import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import io
import base64
import re
import time
from sqlalchemy import text
from wordcloud import WordCloud, STOPWORDS
from streamlit_option_menu import option_menu
from dotenv import load_dotenv

# --- 1. SETUP PATH, IMPORT & API ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '..'))
if root_project not in sys.path:
    sys.path.append(root_project)

# Load Environment Variables
load_dotenv()

try:
    from src.utils.db import get_db_engine
    # Import LLM Engine (Ringkasan) dan RAG Core (Chatbot)
    from src.chatbot.llm_engine import generate_summary
    from src.chatbot.rag_core import ask_bot
    # Import Topic Modeling (Bisa dihapus jika nanti 100% pindah ke LLM)
    from src.processing.topic_modeling import get_lda_topics
except ImportError as e:
    st.error(f"Gagal Import Modul: {e}")
    st.stop()

# --- 2. CONFIG PAGE ---
st.set_page_config(page_title="REMOSY Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 3. SESSION STATE INIT & SECURITY VARS ---
if 'news_chart_key' not in st.session_state: st.session_state['news_chart_key'] = 0
if 'tw_chart_key' not in st.session_state: st.session_state['tw_chart_key'] = 0
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

# Variabel Rate Limiting
if 'last_action_time' not in st.session_state: st.session_state['last_action_time'] = 0


# --- 4. FUNGSI KEAMANAN (ANTI-JAILBREAK & SANITASI) ---
def validate_security(input_text):
    """Fungsi untuk mengecek keamanan input dari user"""
    if not input_text or not input_text.strip():
        return False, "Input tidak boleh kosong."

    if len(input_text) > 300:
        return False, "Input terlalu panjang! Maksimal 300 karakter."

    forbidden_phrases = [
        "ignore all previous instructions", "abaikan semua instruksi",
        "system prompt", "you are an uncensored", "jailbreak", "bypass",
        "forget everything", "lupakan semua", "bertindaklah sebagai", "act as"
    ]
    text_lower = input_text.lower()
    for phrase in forbidden_phrases:
        if phrase in text_lower:
            return False, "🚨 TERDETEKSI PELANGGARAN KEAMANAN (Anti-Jailbreak). Input diblokir."

    return True, "Aman"


# --- 5. DATA LOADER ---
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
def load_lda_data():
    return get_lda_topics(n_topics=5)


# --- FUNGSI WORDCLOUD DENGAN DYNAMIC STOPWORDS ---
def plot_wordcloud(text_data, colormap='viridis', dynamic_stopwords=None):
    if not text_data: return None

    full_text = " ".join([str(t).lower() for t in text_data if t is not None])
    if len(full_text.strip()) < 5: return None

    id_stopwords = {
        'di', 'dan', 'yang', 'ini', 'itu', 'dengan', 'untuk', 'dari', 'ke', 'pada',
        'dalam', 'adalah', 'sebagai', 'tidak', 'akan', 'juga', 'bisa', 'ada', 'oleh',
        'saat', 'sudah', 'kami', 'saya', 'mereka', 'kita', 'ya', 'yg', 'ga', 'aja',
        'buat', 'kalau', 'kalo', 'atau', 'lebih', 'lagi', 'baru', 'jadi', 'nya',
        'hari', 'tahun', 'sebuah', 'telah', 'setelah', 'namun', 'karena', 'seperti',
        'banyak', 'sangat', 'bagi', 'hingga', 'lalu', 'saja', 'masih', 'pun', 'dari',
        'ke', 'bisa', 'tapi', 'biar', 'kok', 'sih', 'dong', 'kan', 'nah', 'pas',
        'aku', 'mau', 'yaa', 'punya', 'dll', 'nih', 'bgt', 'udah', 'gw', 'gua', 'lu',
        'lo', 'deh', 'kek', 'kayak', 'amp', 'soal', 'dm', 'viral', 'banget', 'pls',
        'sama', 'baca', 'cek', 'video', 'info', 'link', 'masuk', 'terima', 'ambil',
        'anak', 'dia', 'kamu', 'ku', 'ft', 'web', 'youtube', 'tiktok', 'instagram'
    }

    custom_stopwords = STOPWORDS.union(id_stopwords)

    custom_stopwords.update([
        'unsika', 'universitas', 'singaperbangsa', 'karawang',
        'mahasiswa', 'kampus', 'fakultas', 'prodi', 'ptn', 'pts',
        'kompasiana', 'tvberita', 'inews', 'infoka', 'antara', 'news',
        'wartakotalive', 'detiknews', 'detik', 'tribun', 'rakyatjelata',
        'karawangnews', 'kumparan', 'radarkarawang', 'media',
        'co', 'id', 'com', 'http', 'https', 't', 'rt', 'www',
        'sbmptnfess', 'sbmptn', 'snbt', 'snmptn', 'utbk', 'umptkin', 'mandiri',
        'suaraunsika', 'unsika_ulcc', 'ulcc', 'maketheuniverseours', 'storeofulcc',
        'ugm', 'ui', 'itb', 'ipb', 'unpad', 'undip', 'unair', 'brawijaya', 'ub',
        'uns', 'uny', 'upi', 'unj', 'unnes', 'usu', 'unhas', 'unand', 'unsri',
        'untan', 'unlam', 'ulm', 'unmul', 'uho', 'unsil', 'udayana', 'unram',
        'upn', 'upnvj', 'upnyk', 'upnjatim', 'binus', 'telkom', 'unkrat',
        'unimed', 'unip', 'unej', 'untid', 'unja', 'unri'
    ])

    if dynamic_stopwords:
        custom_stopwords.update(dynamic_stopwords)

    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=colormap,
            stopwords=custom_stopwords,
            max_words=70,
            collocations=False
        ).generate(full_text)

        img_buffer = io.BytesIO()
        wc.to_image().save(img_buffer, format='PNG')
        return img_buffer
    except Exception:
        return None


def render_image_html(image_buffer):
    b64_img = base64.b64encode(image_buffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64_img}" style="width:100%; border-radius:10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'


def get_color_map():
    return {'Positive': '#00CC96', 'Negative': '#EF553B', 'Neutral': '#636EFA', 'Belum Dinilai': '#D3D3D3'}


# ==========================================
# 🏠 HALAMAN 1: DASHBOARD UTAMA
# ==========================================
def show_overview(df_news, df_tweets):
    st.title("Dashboard Utama")
    st.markdown("Ringkasan eksekutif reputasi kampus dari seluruh kanal.")
    st.divider()

    all_sentiments = pd.concat([df_news['sentiment_label'] if not df_news.empty else pd.Series(),
                                df_tweets['sentiment_label'] if not df_tweets.empty else pd.Series()])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Isu Masuk", len(df_news) + len(df_tweets))
    c2.metric("Sentimen Positif", len(all_sentiments[all_sentiments == 'Positive']), delta="Good")
    c3.metric("Sentimen Negatif", len(all_sentiments[all_sentiments == 'Negative']), delta="-Alert",
              delta_color="inverse")
    c4.metric("Dalam Antrian", len(all_sentiments[all_sentiments == 'Belum Dinilai']))

    st.divider()

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

    c_title, c_btn = st.columns([4, 1])
    with c_title:
        st.subheader("🔥 5 Isu Terhangat")
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


# ==========================================
# 📰 HALAMAN 2: ANALISIS BERITA
# ==========================================
def show_news_analytics(df_news):
    st.title("📰 Analisis Media & Berita")
    st.markdown("Pantau bagaimana media massa memberitakan institusi Anda.")

    if df_news.empty:
        st.warning("Belum ada data berita yang diproses di Database Warehouse.")
        return

    # 1. Barisan Metrik Cepat (KPI)
    st.subheader("Ringkasan Sentimen Media")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Berita", len(df_news))
    c2.metric("Sentimen Positif", len(df_news[df_news['sentiment_label'] == 'Positive']))
    c3.metric("Sentimen Negatif", len(df_news[df_news['sentiment_label'] == 'Negative']))
    c4.metric("Sentimen Netral", len(df_news[df_news['sentiment_label'] == 'Neutral']))

    st.divider()

    # --- FITUR AI: RINGKASAN BERITA ---
    st.subheader("🤖 Ringkasan AI (TL;DR)")
    st.caption("Dihasilkan secara otomatis oleh model Bahasa (LLM) Google Gemini.")

    with st.container(border=True):
        if st.button("✨ Buat Ringkasan Berita Hari Ini", key="btn_ai_news", type="primary"):
            with st.spinner("AI Gemini sedang membaca dan menyimpulkan narasi berita..."):
                try:
                    list_judul_berita = df_news.head(25)['title'].dropna().tolist()
                    ringkasan = generate_summary(list_judul_berita, context_type="Berita Portal")
                    st.success(ringkasan)
                except Exception as e:
                    st.error(f"Gagal memuat ringkasan: {e}")

    st.divider()

    # 2. Grafik Tren & Top 10 Media
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Tren Publikasi Berita Harian")
        trend_news = df_news.groupby(['date', 'sentiment_label']).size().reset_index(name='jumlah')
        trend_news['date'] = pd.to_datetime(trend_news['date'])
        max_date = trend_news['date'].max()
        min_date = trend_news['date'].min()

        fig_trend = px.line(trend_news, x='date', y='jumlah', color='sentiment_label',
                            color_discrete_map=get_color_map(), markers=True)
        fig_trend.update_xaxes(range=[min_date, max_date], rangeselector=dict(
            buttons=list([
                dict(count=1, label="1 Bln", step="month", stepmode="backward"),
                dict(count=3, label="3 Bln", step="month", stepmode="backward"),
                dict(step="all", label="Semua")
            ]), bgcolor="#262730", activecolor="#00CC96", font=dict(color="white")), type="date")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.subheader("🏆 Sentimen per Top 10 Media")
        top_sources_list = df_news['source'].value_counts().head(10).index.tolist()
        df_top_source = df_news[df_news['source'].isin(top_sources_list)]
        source_sentiment = df_top_source.groupby(['source', 'sentiment_label']).size().reset_index(name='jumlah')
        fig_source = px.bar(source_sentiment, x='jumlah', y='source', color='sentiment_label', orientation='h',
                            color_discrete_map=get_color_map(), barmode='stack')
        fig_source.update_yaxes(categoryorder='array', categoryarray=top_sources_list[::-1])
        fig_source.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""))
        st.plotly_chart(fig_source, use_container_width=True)

    st.divider()

    # 3. Analisis Kata Kunci & Pie Chart
    col3, col4 = st.columns([2, 1])
    with col3:
        st.subheader("☁️ Word Cloud Judul Berita")
        media_words = set()
        for source_name in df_news['source'].dropna().unique():
            words = re.findall(r'\b\w+\b', str(source_name).lower())
            media_words.update(words)

        wc_buffer = plot_wordcloud(df_news['title'].dropna().tolist(), colormap='magma', dynamic_stopwords=media_words)
        if wc_buffer:
            st.markdown(render_image_html(wc_buffer), unsafe_allow_html=True)
        else:
            st.info("Data teks belum cukup untuk membentuk pola WordCloud.")

    with col4:
        st.subheader("📊 Proporsi Sentimen")
        sentiment_counts = df_news['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentimen', 'Jumlah']
        fig_pie = px.pie(sentiment_counts, names='Sentimen', values='Jumlah', hole=0.4, color='Sentimen',
                         color_discrete_map=get_color_map())
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # 4. Tabel Data Mentah
    st.subheader("📋 Tabel Data Mentah")
    df_sorted = df_news.sort_values(by='date', ascending=False)
    st.dataframe(
        df_sorted[['date_str', 'source', 'title', 'sentiment_label', 'sentiment_score', 'url']],
        use_container_width=True, hide_index=True, height=400,
        column_config={
            "date_str": "Tanggal", "source": "Sumber", "title": "Judul Berita", "sentiment_label": "Sentimen",
            "sentiment_score": st.column_config.NumberColumn("Skor Sentimen", format="%.2f"),
            "url": st.column_config.LinkColumn("Tautan", display_text="Buka Berita 🔗")
        }
    )


# ==========================================
# 🐦 HALAMAN 3: ANALISIS TWITTER (X)
# ==========================================
def show_twitter_analytics(df_tweets):
    st.title("🐦 Analisis Cuitan X (Twitter)")
    st.markdown("Pantau opini publik dan interaksi warganet terkait institusi Anda.")

    if df_tweets.empty:
        st.warning("Belum ada data cuitan yang diproses di Database Warehouse.")
        return

    # 1. Barisan Metrik Cepat (KPI)
    st.subheader("Ringkasan Sentimen Warganet")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cuitan", len(df_tweets))
    c2.metric("Sentimen Positif", len(df_tweets[df_tweets['sentiment_label'] == 'Positive']))
    c3.metric("Sentimen Negatif", len(df_tweets[df_tweets['sentiment_label'] == 'Negative']))
    c4.metric("Sentimen Netral", len(df_tweets[df_tweets['sentiment_label'] == 'Neutral']))

    st.divider()

    # --- FITUR AI: RINGKASAN TWITTER ---
    st.subheader("🤖 Ringkasan AI (TL;DR)")
    st.caption("Dihasilkan secara otomatis oleh model Bahasa (LLM) Google Gemini.")

    with st.container(border=True):
        if st.button("✨ Buat Ringkasan Cuitan Hari Ini", key="btn_ai_tw", type="primary"):
            with st.spinner("AI Gemini sedang membaca keluh kesah warganet..."):
                try:
                    list_cuitan = df_tweets['clean_text'].dropna().head(25).tolist()
                    ringkasan = generate_summary(list_cuitan, context_type="Cuitan Twitter")
                    st.success(ringkasan)
                except Exception as e:
                    st.error(f"Gagal memuat ringkasan: {e}")

    st.divider()

    # 2. Grafik Tren & Top 10 Akun
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Tren Obrolan Harian")
        trend_tw = df_tweets.groupby(['date', 'sentiment_label']).size().reset_index(name='jumlah')
        trend_tw['date'] = pd.to_datetime(trend_tw['date'])
        max_date = trend_tw['date'].max()
        min_date = trend_tw['date'].min()

        fig_trend = px.line(trend_tw, x='date', y='jumlah', color='sentiment_label',
                            color_discrete_map=get_color_map(), markers=True)
        fig_trend.update_xaxes(range=[min_date, max_date], rangeselector=dict(
            buttons=list([
                dict(count=1, label="1 Bln", step="month", stepmode="backward"),
                dict(count=3, label="3 Bln", step="month", stepmode="backward"),
                dict(step="all", label="Semua")
            ]), bgcolor="#262730", activecolor="#00CC96", font=dict(color="white")), type="date")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.subheader("🏆 Top 10 Akun Paling Aktif")
        top_users_list = df_tweets['username'].value_counts().head(10).index.tolist()
        df_top_users = df_tweets[df_tweets['username'].isin(top_users_list)]
        user_sentiment = df_top_users.groupby(['username', 'sentiment_label']).size().reset_index(name='jumlah')
        fig_user = px.bar(user_sentiment, x='jumlah', y='username', color='sentiment_label', orientation='h',
                          color_discrete_map=get_color_map(), barmode='stack')
        fig_user.update_yaxes(categoryorder='array', categoryarray=top_users_list[::-1])
        fig_user.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""))
        st.plotly_chart(fig_user, use_container_width=True)

    st.divider()

    # 3. Analisis Kata Kunci & Pie Chart
    col3, col4 = st.columns([2, 1])
    with col3:
        st.subheader("☁️ Word Cloud Cuitan")
        user_words = set()
        for username in df_tweets['username'].dropna().unique():
            words = re.findall(r'\b\w+\b', str(username).lower())
            user_words.update(words)

        wc_buffer = plot_wordcloud(df_tweets['clean_text'].dropna().tolist(), colormap='Blues',
                                   dynamic_stopwords=user_words)
        if wc_buffer:
            st.markdown(render_image_html(wc_buffer), unsafe_allow_html=True)
        else:
            st.info("Data teks belum cukup untuk membentuk pola WordCloud.")

    with col4:
        st.subheader("📊 Proporsi Sentimen")
        sentiment_counts = df_tweets['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentimen', 'Jumlah']
        fig_pie = px.pie(sentiment_counts, names='Sentimen', values='Jumlah', hole=0.4, color='Sentimen',
                         color_discrete_map=get_color_map())
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # 4. Tabel Data Mentah
    st.subheader("📋 Tabel Data Mentah")
    df_sorted = df_tweets.sort_values(by='date', ascending=False)
    st.dataframe(
        df_sorted[['date_str', 'username', 'clean_text', 'sentiment_label', 'sentiment_score', 'tweet_url']],
        use_container_width=True, hide_index=True, height=400,
        column_config={
            "date_str": "Tanggal", "username": "Akun (@)", "clean_text": "Isi Cuitan", "sentiment_label": "Sentimen",
            "sentiment_score": st.column_config.NumberColumn("Skor Sentimen", format="%.2f"),
            "tweet_url": st.column_config.LinkColumn("Tautan", display_text="Buka Cuitan 🔗")
        }
    )


# ==========================================
# ⚙ HALAMAN 4: PENGATURAN (DENGAN SECURITY)
# ==========================================
def show_settings():
    st.title("⚙ Pengaturan")
    engine = get_db_engine()

    c1, c2 = st.columns([3, 1])
    new_kw = c1.text_input("Tambah Kata Kunci Baru")

    if c2.button("Simpan", type="primary"):
        current_time = time.time()
        if current_time - st.session_state['last_action_time'] < 2.0:
            st.toast("⏳ Terlalu banyak permintaan! Tunggu 2 detik.", icon="⚠️")
        else:
            st.session_state['last_action_time'] = current_time
            is_safe, msg = validate_security(new_kw)

            if not is_safe:
                st.toast(f"⛔ {msg}", icon="🚨")
            else:
                is_success, is_duplicate = False, False
                try:
                    with engine.begin() as conn:
                        check_df = pd.read_sql(text("SELECT * FROM keywords WHERE keyword = :k"), conn,
                                               params={"k": new_kw})
                        if check_df.empty:
                            conn.execute(text("INSERT INTO keywords (keyword, source) VALUES (:k, 'all')"),
                                         {"k": new_kw})
                            is_success = True
                        else:
                            is_duplicate = True
                except Exception as e:
                    st.toast(f"Gagal menyimpan data!", icon="❌")

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
# 💬 UI FLOATING CHATBOT (DENGAN SECURITY)
# ==========================================
def show_floating_chat():
    if 'chat_expanded' not in st.session_state:
        st.session_state['chat_expanded'] = False

    chat_w = "800px" if st.session_state['chat_expanded'] else "420px"
    chat_h = "85vh" if st.session_state['chat_expanded'] else "600px"
    cont_h = 600 if st.session_state['chat_expanded'] else 400

    st.markdown(
        f"""
        <style>
        div[data-testid="stPopover"] {{ position: fixed !important; bottom: 30px !important; right: 40px !important; z-index: 999999 !important; width: auto !important; }}
        div[data-testid="stPopover"] > button {{ width: 65px !important; height: 65px !important; border-radius: 50% !important; background-color: #00CC96 !important; color: white !important; border: none !important; padding: 0 !important; display: flex !important; justify-content: center !important; align-items: center !important; box-shadow: 0 6px 16px rgba(0,0,0,0.4) !important; transition: transform 0.3s ease; }}
        div[data-testid="stPopover"] > button p {{ font-size: 32px !important; margin: 0 !important; }}
        div[data-testid="stPopover"] > button:hover {{ transform: scale(1.1); }}
        div[data-testid="stPopoverBody"] {{ width: {chat_w} !important; height: {chat_h} !important; max-width: 95vw !important; max-height: 95vh !important; padding: 20px !important; border-radius: 15px !important; border: 1px solid #555 !important; resize: both !important; overflow: auto !important; transition: width 0.3s ease, height 0.3s ease; }}
        </style>
        """, unsafe_allow_html=True
    )

    with st.popover("💬"):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown("### 🤖 Asisten AI")
            st.caption("Data analitik FAISS Vector DB")
        with c2:
            if st.button("⛶" if not st.session_state['chat_expanded'] else "🗕", help="Perbesar / Perkecil Ukuran"):
                st.session_state['chat_expanded'] = not st.session_state['chat_expanded']
                st.rerun()

        st.divider()

        chat_container = st.container(height=cont_h)
        with chat_container:
            if not st.session_state['chat_history']:
                st.markdown(
                    "<div style='text-align:center; color:#888; margin-top:50px;'><i>Ketik pertanyaan di bawah untuk memulai.</i></div>",
                    unsafe_allow_html=True)

            for message in st.session_state['chat_history']:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if st.session_state['chat_history'] and st.session_state['chat_history'][-1]['role'] == 'user':
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Menganalisis..."):
                        try:
                            # --- MEMANGGIL RAG DARI rag_core.py ---
                            answer = ask_bot(st.session_state['chat_history'][-1]['content'])
                            st.session_state['chat_history'].append({"role": "assistant", "content": answer})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal memanggil AI: {e}")

        # --- INPUT DENGAN SECURITY CHECK ---
        if prompt := st.chat_input("Tanyakan isu reputasi kampus..."):
            current_time = time.time()
            if current_time - st.session_state['last_action_time'] < 3.0:
                st.toast("⏳ Dimohon tidak melakukan spam. Tunggu 3 detik.", icon="⚠️")
            else:
                st.session_state['last_action_time'] = current_time
                is_safe, msg = validate_security(prompt)

                if not is_safe:
                    st.toast(msg, icon="🚨")
                    st.session_state['chat_history'].append({"role": "user", "content": prompt})
                    st.session_state['chat_history'].append(
                        {"role": "assistant", "content": f"🛡️ **Sistem Keamanan REMOSY:** Akses ditolak. {msg}"})
                    st.rerun()
                else:
                    st.session_state['chat_history'].append({"role": "user", "content": prompt})
                    st.rerun()


# ==========================================
# 🚀 MAIN APP DENGAN MODERN SIDEBAR
# ==========================================
def main():
    df_news, df_tweets = load_all_data()

    with st.sidebar:
        st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'> Reputation Monitoring System</h1>",
                    unsafe_allow_html=True)
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

    show_floating_chat()


if __name__ == "__main__":
    main()