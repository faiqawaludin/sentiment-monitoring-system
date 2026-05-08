import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import io
import base64
from sqlalchemy import text
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
import time
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai

# --- 1. SETUP PATH, IMPORT & API ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '..'))
if root_project not in sys.path: sys.path.append(root_project)

try:
    from src.utils.db import get_db_engine
    from src.processing.topic_modeling import get_lda_topics
except ImportError as e:
    st.error(f"Gagal Import Modul: {e}")
    st.stop()

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- 2. CONFIG PAGE ---
st.set_page_config(page_title="REMOSY Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 3. SESSION STATE INIT ---
if 'news_chart_key' not in st.session_state: st.session_state['news_chart_key'] = 0
if 'tw_chart_key' not in st.session_state: st.session_state['tw_chart_key'] = 0
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []


# --- 4. DATA LOADER ---
@st.cache_data(ttl=60)
def load_all_data():
    engine = get_db_engine()
    q_news = """SELECT nr.id, \
                       nr.title, \
                       nr.source, \
                       nr.url, \
                       nr.scraped_at, \
                       nr.published_date,
                       COALESCE(np.sentiment_label, 'Belum Dinilai') as sentiment_label,
                       COALESCE(np.sentiment_score, 0)               as sentiment_score
                FROM news_raw nr \
                         LEFT JOIN news_processed np ON nr.id = np.news_id
                ORDER BY nr.published_date DESC NULLS LAST"""
    q_tweets = """SELECT tr.id, \
                         tr.username, \
                         tr.full_text                                  as clean_text, \
                         tr.tweet_url, \
                         tr.scraped_at,
                         COALESCE(tp.sentiment_label, 'Belum Dinilai') as sentiment_label,
                         COALESCE(tp.sentiment_score, 0)               as sentiment_score
                  FROM tweets_raw tr \
                           LEFT JOIN tweets_processed tp ON tr.id = tp.tweet_id
                  ORDER BY tr.scraped_at DESC"""
    try:
        df_news = pd.read_sql(q_news, engine)
        df_tweets = pd.read_sql(q_tweets, engine)
        if not df_news.empty:
            df_news['published_date'] = pd.to_datetime(df_news['published_date'], errors='coerce')
            df_news['scraped_at'] = pd.to_datetime(df_news['scraped_at'], errors='coerce')
            df_news['final_date'] = df_news['published_date'].fillna(df_news['scraped_at'])
            df_news['date'] = df_news['final_date'].dt.date
            df_news['date_str'] = df_news['final_date'].dt.strftime('%d %b %Y')
        if not df_tweets.empty:
            df_tweets['scraped_at'] = pd.to_datetime(df_tweets['scraped_at'], errors='coerce')
            df_tweets['date'] = df_tweets['scraped_at'].dt.date
            df_tweets['date_str'] = df_tweets['scraped_at'].dt.strftime('%d %b %Y %H:%M')
        return df_news, df_tweets
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def load_lda_data(): return get_lda_topics(n_topics=5)


def plot_wordcloud(text_data, colormap='viridis'):
    if not text_data: return None
    full_text = " ".join([str(t) for t in text_data if t is not None])
    if len(full_text.strip()) < 5: return None
    try:
        wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(full_text)
        img_buffer = io.BytesIO()
        wc.to_image().save(img_buffer, format='PNG')
        return img_buffer
    except Exception:
        return None


def render_image_html(image_buffer):
    b64_img = base64.b64encode(image_buffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64_img}" style="width:100%; border-radius:10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'


def get_color_map(): return {'Positive': '#00CC96', 'Negative': '#EF553B', 'Neutral': '#636EFA',
                             'Belum Dinilai': '#D3D3D3'}


# ==========================================
# 🏠 HALAMAN 1: DASHBOARD UTAMA
# ==========================================
def show_overview(df_news, df_tweets):
    st.title("Dashboard Utama")
    st.markdown("Ringkasan eksekutif reputasi kampus dari seluruh kanal.")
    st.divider()

    # --- Metrics ---
    all_sentiments = pd.concat([df_news['sentiment_label'] if not df_news.empty else pd.Series(),
                                df_tweets['sentiment_label'] if not df_tweets.empty else pd.Series()])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Isu Masuk", len(df_news) + len(df_tweets))
    c2.metric("Sentimen Positif", len(all_sentiments[all_sentiments == 'Positive']), delta="Good")
    c3.metric("Sentimen Negatif", len(all_sentiments[all_sentiments == 'Negative']), delta="-Alert",
              delta_color="inverse")
    c4.metric("Dalam Antrian", len(all_sentiments[all_sentiments == 'Belum Dinilai']))

    st.divider()

    # --- Tren Isu & Proporsi ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Tren Isu Harian")
        trend_frames = []
        if not df_news.empty: trend_frames.append(df_news[['date', 'sentiment_label']])
        if not df_tweets.empty: trend_frames.append(df_tweets[['date', 'sentiment_label']])
        if trend_frames:
            full_trend = pd.concat(trend_frames).groupby(['date', 'sentiment_label']).size().reset_index(name='jumlah')
            fig_trend = px.bar(full_trend, x='date', y='jumlah', color='sentiment_label', barmode='stack',
                               color_discrete_map=get_color_map())
            max_date = pd.to_datetime(full_trend['date']).max()
            min_date = max_date - pd.Timedelta(days=30)
            fig_trend.update_xaxes(range=[min_date, max_date], rangeselector=dict(buttons=list([
                dict(count=1, label="1 Bulan", step="month", stepmode="backward"),
                dict(count=3, label="3 Bulan", step="month", stepmode="backward"),
                dict(step="all", label="Semua Data")
            ]), bgcolor="#1E1E1E", activecolor="#00CC96"), type="date")
            st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.subheader("Proporsi Global")
        if not all_sentiments.empty:
            pie_data = all_sentiments.value_counts().reset_index()
            pie_data.columns = ['Sentimen', 'Jumlah']
            fig_pie = px.pie(pie_data, names='Sentimen', values='Jumlah', hole=0.4, color='Sentimen',
                             color_discrete_map=get_color_map())
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # --- TOPIK LDA ---
    c_title, c_btn = st.columns([4, 1])
    with c_title:
        st.subheader("🔥 5 Isu Terhangat (AI Analysis)")
    with c_btn:
        if st.button("🔄 Refresh Topik"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("🤖 AI sedang membaca & menyimpulkan isu kampus..."):
        topics_data = load_lda_data()

    if isinstance(topics_data, dict) and "error" in topics_data:
        st.error("Gagal memuat topik.")
    elif not topics_data:
        st.warning("Data belum cukup untuk analisis topik.")
    else:
        cols = st.columns(5)
        for i, topic in enumerate(topics_data):
            with cols[i % 5]:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 20px; border-radius: 12px; border-top: 4px solid #FF4B4B; height: 180px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); display: flex; flex-direction: column; justify-content: space-between;">
                    <h4 style="margin:0 0 10px 0; color: white; font-size: 15px; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;">{topic['title']}</h4>
                    <div style="border-top: 1px solid #444; padding-top: 10px;">
                        <p style="font-size: 11px; color: #888; margin: 0;">🗝️ <b>Keywords:</b></p>
                        <p style="font-size: 11px; color: #AAA; font-style: italic; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{', '.join(topic['keywords'][:4])}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("💡 Penjelasan AI & Bukti Berita"):
                    if topic.get('summary') and topic['summary'] != "Tidak ada ringkasan.":
                        st.markdown(f"**Ringkasan:** {topic['summary']}")
                    st.info(topic.get('detail', 'Penjelasan rinci dari AI tidak tersedia.'))
                    if topic['keywords']:
                        pattern = '|'.join(topic['keywords'][:3])
                        related_news = df_news[df_news['title'].str.contains(pattern, case=False, na=False)].head(2)
                        if not related_news.empty:
                            st.markdown("**📰 Sumber Berita:**")
                            for _, r in related_news.iterrows():
                                st.markdown(f"- [{r['title'][:50]}...]({r['url']})")
                    else:
                        st.caption("Tidak ada berita terkait.")

    st.divider()

    # --- 💬 FITUR CHATBOT INLINE TANYA AI (FIXED ATTACHMENT) ---
    st.subheader("💬 Asisten AI REMOSY")
    st.caption(
        "Tanyakan apa saja seputar reputasi kampus hari ini! (Contoh: 'Tolong buatkan draf siaran pers untuk menanggapi isu negatif hari ini')")

    # KUNCI SOLUSI: Menggunakan 1 Container induk agar input chat menempel pada kotak obrolan, bukan dasar layar
    chat_wrapper = st.container()

    with chat_wrapper:
        # 1. Bagian riwayat obrolan (Fixed Height)
        chat_container = st.container(height=350)
        with chat_container:
            if not st.session_state['chat_history']:
                st.markdown(
                    "<div style='text-align:center; color:#888; margin-top:100px;'><i>Belum ada obrolan. Silakan ketik pertanyaan di bawah.</i></div>",
                    unsafe_allow_html=True)
            for message in st.session_state['chat_history']:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 2. Bagian input (Akan merapat rapi tepat di bawah riwayat obrolan)
        if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
            st.session_state['chat_history'].append({"role": "user", "content": prompt})
            st.rerun()

    st.divider()

    # --- SOROTAN NEGATIF ---
    st.subheader("🚨 Sorotan Negatif (Prioritas Tinggi)")

    neg_news = df_news[df_news['sentiment_label'] == 'Negative'].sort_values(by=['final_date', 'sentiment_score'],
                                                                             ascending=[False, False])
    neg_tweets = df_tweets[df_tweets['sentiment_label'] == 'Negative'].sort_values(by=['scraped_at', 'sentiment_score'],
                                                                                   ascending=[False, False])

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.markdown("### 📰 Berita Negatif")
        if not neg_news.empty:
            with st.container(height=500):
                for _, row in neg_news.iterrows():
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; border-left: 5px solid #EF553B; background-color: #262730; margin-bottom: 10px;">
                        <h4 style="margin:0; font-size: 16px;"><a href="{row['url']}" target="_blank" style="text-decoration:none; color: #4DA6FF;">{row['title']} 🔗</a></h4>
                        <div style="margin-top: 5px; font-size: 12px; color: #AAAAAA;">📅 <b>{row['date_str']}</b> | 📰 {row['source']}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.success("Aman. Tidak ada berita negatif.")

    with col_n2:
        st.markdown("### 🐦 Tweet Negatif")
        if not neg_tweets.empty:
            with st.container(height=500):
                for _, row in neg_tweets.iterrows():
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; border-left: 5px solid #EF553B; background-color: #262730; margin-bottom: 10px;">
                        <div style="font-size: 14px; color: #FAFAFA; margin-bottom: 5px;">"{row['clean_text']}"</div>
                        <div style="font-size: 12px; color: #AAAAAA;">👤 <b>@{row['username']}</b> | 📅 {row['date_str']}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.success("Aman. Tidak ada tweet negatif.")


# --- PROSES CHATBOT AI ---
if st.session_state.get('chat_history') and st.session_state['chat_history'][-1]['role'] == 'user':
    df_news, df_tweets = load_all_data()
    neg_news = df_news[df_news['sentiment_label'] == 'Negative'].head(3)['title'].tolist() if not df_news.empty else []
    pos_news = df_news[df_news['sentiment_label'] == 'Positive'].head(3)['title'].tolist() if not df_news.empty else []

    system_context = f"""
    Kamu adalah Asisten AI untuk Sistem Monitoring Reputasi Kampus (REMOSY) Universitas Singaperbangsa Karawang.
    Konteks Hari Ini: Berita Positif ({pos_news}), Berita Negatif ({neg_news}).
    Pertanyaan User: {st.session_state['chat_history'][-1]['content']}
    """

    if ai_client:
        try:
            response = ai_client.models.generate_content(model="gemini-flash-latest", contents=system_context)
            st.session_state['chat_history'].append({"role": "assistant", "content": response.text})
            st.rerun()
        except Exception as e:
            st.session_state['chat_history'].append({"role": "assistant", "content": "⚠️ Gagal menghubungi AI."})
            st.rerun()
    else:
        st.session_state['chat_history'].append({"role": "assistant", "content": "⚠️ API Key belum diatur."})
        st.rerun()


# ==========================================
# 📰 HALAMAN 2: ANALISIS BERITA (RESTORED FULL)
# ==========================================
def show_news_analytics(df_news):
    st.title("Analisis Media & Berita")
    if df_news.empty: st.warning("Belum ada data."); return

    total_news = len(df_news)
    pos_news = len(df_news[df_news['sentiment_label'] == 'Positive'])
    neg_news = len(df_news[df_news['sentiment_label'] == 'Negative'])
    neu_news = len(df_news[df_news['sentiment_label'] == 'Neutral'])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Berita", total_news)
    m2.metric("Positif", pos_news, delta=f"{pos_news / total_news:.1%}" if total_news else "0%")
    m3.metric("Negatif", neg_news, delta=f"{neg_news / total_news:.1%}" if total_news else "0%", delta_color="inverse")
    m4.metric("Netral", neu_news)

    st.divider()
    st.subheader("🏆 Top 5 Media Teraktif")

    media_stats = df_news.groupby('source').agg(jumlah_artikel=('id', 'count'),
                                                avg_score=('sentiment_score', 'mean')).sort_values(by='jumlah_artikel',
                                                                                                   ascending=False).head(
        5)

    if not media_stats.empty:
        cols = st.columns(5)
        media_list = media_stats.reset_index().to_dict('records')
        for i in range(5):
            with cols[i]:
                if i < len(media_list):
                    row = media_list[i]
                    color = "#00CC96" if row['avg_score'] > 0.05 else "#EF553B" if row[
                                                                                       'avg_score'] < -0.05 else "#636EFA"
                    status = "Positif" if row['avg_score'] > 0.05 else "Negatif" if row[
                                                                                        'avg_score'] < -0.05 else "Netral"
                    html_card = f"""<div style="border-top: 5px solid {color}; background-color: #262730; padding: 15px; border-radius: 8px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                    <div style="font-size: 12px; color: #AAAAAA; font-weight: bold;">{status}</div>
                    <div style="font-size: 15px; color: white; font-weight: 500; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis;" title="{row['source']}">{row['source']}</div>
                    <div><span style="font-size: 32px; font-weight:bold; color: {color};">{row['jumlah_artikel']}</span><span style="font-size: 12px; color: #AAAAAA;">Artikel</span></div>
                    </div>"""
                    st.markdown(html_card, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="height: 160px;"></div>', unsafe_allow_html=True)

    st.divider()

    sel_date, sel_source, sel_sent = None, None, None
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.subheader("📈 Tren Berita dari Waktu ke Waktu")
        daily_trend = df_news.groupby(['date', 'sentiment_label']).size().reset_index(name='jumlah')
        if not daily_trend.empty:
            fig_trend = px.bar(daily_trend, x='date', y='jumlah', color='sentiment_label',
                               color_discrete_map=get_color_map())
            df_total_harian = daily_trend.groupby('date')['jumlah'].sum().reset_index().sort_values(by='date')
            fig_trend.add_trace(go.Scatter(x=df_total_harian['date'], y=df_total_harian['jumlah'], mode='lines',
                                           line=dict(color='#FFD700', width=3, shape='spline'), showlegend=False))
            fig_trend.update_layout(xaxis_title=None, legend_title=None, clickmode='event+select')
            event_trend = st.plotly_chart(fig_trend, use_container_width=True, on_select="rerun",
                                          key=f"trend_chart_{st.session_state['news_chart_key']}")
            if event_trend and event_trend.get("selection", {}).get("points"):
                pt = event_trend["selection"]["points"][0]
                sel_date = pt.get("x")
                sel_sent_temp = pt.get("legendgroup")
                if sel_sent_temp and sel_sent_temp != 'Total Tren': sel_sent = sel_sent_temp

    with c_chart2:
        st.subheader("📊 Pola Sentimen per Media")
        media_sentiment = df_news.groupby(['source', 'sentiment_label']).size().reset_index(name='jumlah')
        if not media_sentiment.empty:
            fig_media = px.bar(media_sentiment, x='source', y='jumlah', color='sentiment_label', barmode='stack',
                               color_discrete_map=get_color_map())
            fig_media.update_layout(xaxis_title=None, legend_title=None, clickmode='event+select')
            event_media = st.plotly_chart(fig_media, use_container_width=True, on_select="rerun",
                                          key=f"media_chart_{st.session_state['news_chart_key']}")
            if event_media and event_media.get("selection", {}).get("points"):
                pt = event_media["selection"]["points"][0]
                sel_source = pt.get("x")
                sel_sent_temp = pt.get("legendgroup")
                if sel_sent_temp: sel_sent = sel_sent_temp

    df_display = df_news.copy()
    if sel_date or sel_source or sel_sent:
        filters_text = []
        if sel_date:
            filters_text.append(f"📅 Tanggal: **{sel_date}**")
            df_display = df_display[df_display['date'].astype(str) == str(sel_date)]
        if sel_source:
            filters_text.append(f"📰 Media: **{sel_source}**")
            df_display = df_display[df_display['source'] == sel_source]
        if sel_sent:
            filters_text.append(f"🎭 Sentimen: **{sel_sent}**")
            df_display = df_display[df_display['sentiment_label'] == sel_sent]

        col_f1, col_f2 = st.columns([4, 1])
        with col_f1:
            st.success(f"🔍 Filter Aktif: {' | '.join(filters_text)}")
        with col_f2:
            if st.button("Hapus Semua Filter", type="primary"):
                st.session_state['news_chart_key'] += 1
                st.rerun()

    st.divider()
    st.subheader(f"📋 Data Terpilih: {len(df_display)} Artikel")

    c1, c2 = st.columns([2, 1])
    with c1:
        sort_by = st.selectbox("Urutkan:", ["Tanggal Terbit", "Skor Sentimen", "Nama Media"], key="sb_news")
    with c2:
        sort_order = st.radio("Arah:", ["Baru/Tinggi", "Lama/Rendah"], horizontal=True, key="so_news")

    asc = True if "Lama" in sort_order else False
    if sort_by == "Tanggal Terbit":
        df_display = df_display.sort_values(by='final_date', ascending=asc)
    elif sort_by == "Skor Sentimen":
        df_display = df_display.sort_values(by='sentiment_score', ascending=asc)
    elif sort_by == "Nama Media":
        df_display = df_display.sort_values(by='source', ascending=asc)

    st.dataframe(df_display[['date_str', 'source', 'title', 'sentiment_label', 'sentiment_score', 'url']],
                 column_config={"url": st.column_config.LinkColumn("Link", display_text="🔗 Buka"),
                                "sentiment_score": st.column_config.NumberColumn("Skor", format="%.2f")},
                 use_container_width=True, hide_index=True, height=400)

    st.divider()
    st.subheader("☁️ Topik Populer (Berdasarkan Filter)")
    wc_img = plot_wordcloud(df_display['title'].tolist())
    if wc_img:
        wc_c1, wc_c2, wc_c3 = st.columns([1, 8, 1])
        with wc_c2: st.markdown(render_image_html(wc_img), unsafe_allow_html=True)


# ==========================================
# 🐦 HALAMAN 3: ANALISIS TWITTER (RESTORED FULL)
# ==========================================
def show_twitter_analytics(df_tweets):
    st.title("Analisis Cuitan X")
    if df_tweets.empty: st.warning("Belum ada data."); return

    ct1, ct2, ct3 = st.columns(3)
    ct1.metric("Total Tweet", len(df_tweets))
    ct2.metric("Positif", len(df_tweets[df_tweets['sentiment_label'] == 'Positive']))
    ct3.metric("Negatif", len(df_tweets[df_tweets['sentiment_label'] == 'Negative']))
    st.divider()

    col_t1, col_t2 = st.columns([2, 1])
    sel_user, sel_sent = None, None

    with col_t1:
        st.subheader("Akun Paling Aktif")
        active_tweets = df_tweets[df_tweets['username'] != 'unknown']
        top_users = active_tweets['username'].value_counts().head(10).index
        user_sentiment = active_tweets[active_tweets['username'].isin(top_users)].groupby(
            ['username', 'sentiment_label']).size().reset_index(name='jumlah')

        if not user_sentiment.empty:
            fig_user = px.bar(user_sentiment, x='jumlah', y='username', orientation='h', color='sentiment_label',
                              color_discrete_map=get_color_map())
            event = st.plotly_chart(fig_user, use_container_width=True, on_select="rerun",
                                    key=f"tw_chart_{st.session_state['tw_chart_key']}")
            if event and event["selection"]["points"]:
                sel_user = event["selection"]["points"][0]["y"]
                sel_sent = event["selection"]["points"][0].get("legendgroup")
                st.info(f"Filter: User **@{sel_user}** | Sentimen **{sel_sent}**")
                if st.button("Reset Filter", type="primary"):
                    st.session_state['tw_chart_key'] += 1
                    st.rerun()

    with col_t2:
        st.subheader("Sentimen Netizen")
        pie_data = df_tweets['sentiment_label'].value_counts().reset_index(name='jumlah')
        fig_pie = px.pie(pie_data, names='sentiment_label', values='jumlah', hole=0.4, color='sentiment_label',
                         color_discrete_map=get_color_map())
        st.plotly_chart(fig_pie, use_container_width=True)

    df_display = df_tweets.copy()
    if sel_user: df_display = df_display[df_display['username'] == sel_user]
    if sel_sent: df_display = df_display[df_display['sentiment_label'] == sel_sent]

    st.divider()
    st.subheader(f"📋 Data Terpilih: {len(df_display)} Tweet")
    st.dataframe(df_display[['date_str', 'username', 'clean_text', 'sentiment_label']], use_container_width=True,
                 height=400)


# ==========================================
# ⚙ HALAMAN 4: PENGATURAN (RESTORED FULL)
# ==========================================
def show_settings():
    st.title("⚙ Pengaturan")
    engine = get_db_engine()

    c1, c2 = st.columns([3, 1])
    new_kw = c1.text_input("Tambah Kata Kunci Baru")

    if c2.button("Simpan", type="primary"):
        if new_kw:
            is_success, is_duplicate = False, False
            try:
                with engine.begin() as conn:
                    check_df = pd.read_sql(text("SELECT * FROM keywords WHERE keyword = :k"), conn,
                                           params={"k": new_kw})
                    if check_df.empty:
                        conn.execute(text("INSERT INTO keywords (keyword, source) VALUES (:k, 'all')"), {"k": new_kw})
                        is_success = True
                    else:
                        is_duplicate = True
            except Exception as e:
                st.toast(f"Gagal menyimpan data! Cek log error.", icon="❌")

            if is_success:
                st.toast(f"Berhasil menambahkan keyword: {new_kw}", icon="✅")
                time.sleep(1.2)
                st.rerun()
            elif is_duplicate:
                st.toast(f"Keyword '{new_kw}' sudah ada!", icon="⚠️")

    st.subheader("Daftar Keyword Aktif")
    try:
        df_kw = pd.read_sql("SELECT * FROM keywords ORDER BY id DESC", engine)
        for _, row in df_kw.iterrows():
            col_a, col_b = st.columns([4, 1])
            col_a.text(f"• {row['keyword']}")
            if col_b.button("Hapus", key=f"del_{row['id']}"):
                try:
                    with engine.begin() as conn:
                        conn.execute(text("DELETE FROM keywords WHERE id=:id"), {"id": row['id']})
                    st.toast(f"Keyword '{row['keyword']}' berhasil dihapus!", icon="🗑️")
                    time.sleep(1.2)
                    st.rerun()
                except Exception as e:
                    st.toast(f"Gagal menghapus data!", icon="❌")
    except Exception as e:
        st.error(f"Gagal memuat daftar keyword: {e}")


# ==========================================
# 🚀 MAIN APP DENGAN MODERN SIDEBAR
# ==========================================
def main():
    df_news, df_tweets = load_all_data()

    with st.sidebar:
        st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>🦅 REMOSY</h1>", unsafe_allow_html=True)
        selected = option_menu(
            menu_title=None,
            options=["Utama", "Berita (Media)", "X (Cuitan)", "Pengaturan"],
            icons=["house", "newspaper", "twitter", "gear"],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#262730"},
                "icon": {"color": "#FAFAFA", "font-size": "16px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#00CC96"},
            }
        )
        st.markdown("---")
        st.caption("🤖 Diperbarui otomatis setiap hari.")

    if selected == "Utama":
        show_overview(df_news, df_tweets)
    elif selected == "Berita (Media)":
        show_news_analytics(df_news)
    elif selected == "X (Cuitan)":
        show_twitter_analytics(df_tweets)
    elif selected == "Pengaturan":
        show_settings()


if __name__ == "__main__":
    main()