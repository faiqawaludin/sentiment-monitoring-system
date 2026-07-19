import sys
import os
import io
import base64
import re
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from wordcloud import WordCloud, STOPWORDS
from streamlit_option_menu import option_menu
from dotenv import load_dotenv

# --- 1. SETUP PATH & IMPORT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '..'))
if root_project not in sys.path:
    sys.path.append(root_project)

load_dotenv()

try:
    from src.utils.db import get_db_engine
    from src.chatbot.rag_core import ask_bot
    from src.chatbot.date_filter import filter_recent_news, filter_recent_tweets
except ImportError as e:
    st.error(f"Gagal Import Modul: {e}")
    st.stop()

st.set_page_config(page_title="REMOSY Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- INJEKSI CSS KELAS ENTERPRISE (POWER BI / TABLEAU 3D STYLE) ---
st.markdown("""
<style>
    /* 1. Latar Belakang Kanvas (Abu-abu di Light Mode, Hitam di Dark Mode) */
    [data-testid="stAppViewContainer"] {
        background-color: var(--secondary-background-color); 
    }
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* 2. Panel Kartu 3D Melayang (st.container border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--primary-background-color) !important;
        border-radius: 12px;
        border: none !important; /* Hilangkan garis kaku */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important; /* Efek 3D Bayangan */
        padding: 15px !important;
        margin-bottom: 5px;
    }

    /* 3. KOTAK KPI METRIC (TINGGI DIKUNCI MUTLAK BIAR RATA!) */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 10px;
        padding: 15px 20px;
        height: 130px !important; /* KUNCI MUTLAK TINGGI KARTU */
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    /* Mengunci ruang kosong di dalam kartu KPI agar kontennya tidak lari */
    div[data-testid="stMetricLabel"] { height: 25px; }
    div[data-testid="stMetricValue"] { height: 40px; }
    div[data-testid="stMetricDelta"] { min-height: 20px; }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        border-color: #00CC96;
    }

    /* 4. Pembatas */
    hr {
        margin: 1.5em 0 !important;
        opacity: 0.15;
    }
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'last_action_time' not in st.session_state: st.session_state['last_action_time'] = 0


def validate_security(input_text):
    if not input_text or not input_text.strip(): return False, "Input tidak boleh kosong."
    if len(input_text) > 700: return False, "Input terlalu panjang. Maksimal 700 karakter."
    forbidden = ["ignore all", "abaikan", "system prompt", "uncensored", "jailbreak", "bypass", "act as"]
    for phrase in forbidden:
        if phrase in input_text.lower(): return False, "TERDETEKSI PELANGGARAN KEAMANAN."
    return True, "Aman"


@st.cache_data(ttl=60)
def load_ai_cache():
    engine = get_db_engine()
    try:
        q = "SELECT * FROM ai_analysis_cache ORDER BY updated_at DESC LIMIT 1"
        df_ai = pd.read_sql(q, engine)
        if not df_ai.empty:
            return df_ai.iloc[0]
        return None
    except Exception:
        return None


@st.cache_data(ttl=300)
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


def plot_wordcloud(text_data, colormap='viridis', dynamic_stopwords=None):
    if not text_data: return None
    full_text = " ".join([str(t).lower() for t in text_data if t is not None])
    if len(full_text.strip()) < 5: return None
    id_stopwords = {'di', 'dan', 'yang', 'ini', 'itu', 'dengan', 'untuk', 'dari', 'ke', 'pada', 'dalam', 'adalah',
                    'sebagai', 'tidak', 'akan', 'juga', 'bisa', 'ada', 'oleh', 'saat', 'sudah', 'kami', 'saya',
                    'mereka', 'kita', 'ya', 'yg', 'ga', 'aja', 'buat', 'kalau', 'kalo', 'atau', 'lebih', 'lagi', 'baru',
                    'jadi', 'nya', 'hari', 'tahun', 'sebuah', 'telah', 'setelah', 'namun', 'karena', 'seperti',
                    'banyak', 'sangat', 'bagi', 'hingga', 'lalu', 'saja', 'masih', 'pun', 'tapi', 'biar', 'kok', 'sih',
                    'dong', 'kan', 'nah', 'pas', 'aku', 'mau', 'yaa', 'punya', 'dll', 'nih', 'bgt', 'udah', 'gw', 'gua',
                    'lu', 'lo', 'deh', 'kek', 'kayak', 'amp', 'soal', 'dm', 'viral', 'banget', 'pls', 'sama', 'baca',
                    'cek', 'video', 'info', 'link', 'masuk', 'terima', 'ambil', 'anak', 'dia', 'kamu', 'ku', 'ft',
                    'web', 'youtube', 'tiktok', 'instagram'}
    custom_stopwords = STOPWORDS.union(id_stopwords)
    custom_stopwords.update(
        ['unsika', 'universitas', 'singaperbangsa', 'karawang', 'mahasiswa', 'kampus', 'fakultas', 'prodi', 'ptn',
         'pts', 'kompasiana', 'tvberita', 'inews', 'infoka', 'antara', 'news', 'wartakotalive', 'detiknews', 'detik',
         'tribun', 'rakyatjelata', 'karawangnews', 'kumparan', 'radarkarawang', 'media', 'co', 'id', 'com', 'http',
         'https', 't', 'rt', 'www', 'sbmptnfess', 'sbmptn', 'snbt', 'snmptn', 'utbk', 'umptkin', 'mandiri',
         'suaraunsika', 'unsika_ulcc', 'ulcc', 'maketheuniverseours', 'storeofulcc'])
    if dynamic_stopwords: custom_stopwords.update(dynamic_stopwords)
    try:
        wc = WordCloud(width=800, height=400, background_color='rgba(255, 255, 255, 0)', colormap=colormap,
                       stopwords=custom_stopwords,
                       max_words=70, collocations=False).generate(full_text)
        img_buffer = io.BytesIO()
        wc.to_image().save(img_buffer, format='PNG')
        return img_buffer
    except Exception:
        return None


def render_image_html(image_buffer):
    b64_img = base64.b64encode(image_buffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64_img}" style="width:100%; border-radius:8px;">'


def get_color_map(): return {'Positive': '#00CC96', 'Negative': '#EF553B', 'Neutral': '#636EFA',
                             'Belum Dinilai': '#D3D3D3'}


def render_tldr_ui(ringkasan_text, title="Insight Eksekutif AI"):
    with st.expander(f"**{title}**", expanded=True):
        st.markdown(f"""
        <div style="background-color: rgba(128, 128, 128, 0.05); padding: 20px; border-radius: 8px; border-left: 4px solid #00CC96; border: 1px solid rgba(128, 128, 128, 0.1); margin-bottom: 5px;">
            <div style="font-size: 15px; line-height: 1.6;">
                {ringkasan_text}
            </div>
        </div>
        """, unsafe_allow_html=True)


def filter_by_time(df, date_col, time_filter):
    if df.empty or time_filter == "Semua Waktu":
        return df
    dates = pd.to_datetime(df[date_col], errors='coerce')
    if dates.empty or dates.isna().all():
        return df
    max_date = dates.max()
    if time_filter == "1 Bulan":
        cutoff = max_date - pd.DateOffset(months=1)
    elif time_filter == "3 Bulan":
        cutoff = max_date - pd.DateOffset(months=3)
    elif time_filter == "6 Bulan":
        cutoff = max_date - pd.DateOffset(months=6)
    elif time_filter == "1 Tahun":
        cutoff = max_date - pd.DateOffset(years=1)
    else:
        return df
    return df[dates >= cutoff]


# ==========================================
# HALAMAN 1: DASHBOARD UTAMA
# ==========================================
def show_overview(df_news, df_tweets, ai_cache):
    st.title("Dashboard Utama")
    st.markdown("Ringkasan eksekutif reputasi institusi dari seluruh kanal.")
    st.divider()

    all_sentiments = pd.concat([df_news['sentiment_label'] if not df_news.empty else pd.Series(),
                                df_tweets['sentiment_label'] if not df_tweets.empty else pd.Series()])

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Isu Masuk", len(df_news) + len(df_tweets), delta=" ", delta_color="off")
        c2.metric("Sentimen Positif", len(all_sentiments[all_sentiments == 'Positive']), delta="Good")
        c3.metric("Sentimen Negatif", len(all_sentiments[all_sentiments == 'Negative']), delta="-Alert",
                  delta_color="inverse")
        c4.metric("Dalam Antrian", len(all_sentiments[all_sentiments == 'Belum Dinilai']), delta=" ", delta_color="off")

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            st.subheader("Tren Isu Harian")
            time_opt = st.radio("Rentang Waktu", ["1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun", "Semua Waktu"],
                                index=0, horizontal=True, label_visibility="collapsed", key="time_overview")

            trend_frames = []
            if not df_news.empty:
                df_n_trend = filter_by_time(df_news, 'final_date', time_opt)
                trend_frames.append(df_n_trend[['date', 'sentiment_label']])
            if not df_tweets.empty:
                df_t_trend = filter_by_time(df_tweets, 'scraped_at', time_opt)
                trend_frames.append(df_t_trend[['date', 'sentiment_label']])

            if trend_frames:
                full_trend = pd.concat(trend_frames).groupby(['date', 'sentiment_label']).size().reset_index(
                    name='jumlah')
                fig_trend = px.bar(full_trend, x='date', y='jumlah', color='sentiment_label', barmode='stack',
                                   color_discrete_map=get_color_map())
                # Bikin chart background transparan biar nyatu sama panel
                fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Proporsi Global")
            if not all_sentiments.empty:
                pie_data = all_sentiments.value_counts().reset_index()
                pie_data.columns = ['Sentimen', 'Jumlah']
                fig_pie = px.pie(pie_data, names='Sentimen', values='Jumlah', hole=0.4, color='Sentimen',
                                 color_discrete_map=get_color_map())
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    c_title, c_btn = st.columns([4, 1])
    with c_title:
        st.subheader("5 Isu Terhangat (AI Analysis)")
    with c_btn:
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Mengambil hasil AI..."):
        if ai_cache is not None and ai_cache.get('hot_topics'):
            update_time = pd.to_datetime(ai_cache['updated_at']).strftime('%d %b %Y, %H:%M')
            render_tldr_ui(ai_cache['hot_topics'], title=f"Analisis 5 Isu Terhangat (Update: {update_time})")
        else:
            st.info(
                "AI sedang menyusun analisis. Silakan tunggu jadwal Airflow berjalan atau pastikan Airflow sudah selesai.")

    st.divider()
    st.subheader("Sorotan Negatif (Prioritas Tinggi)")
    neg_news = df_news[df_news['sentiment_label'] == 'Negative'].sort_values(by=['final_date', 'sentiment_score'],
                                                                             ascending=[False, False])
    neg_tweets = df_tweets[df_tweets['sentiment_label'] == 'Negative'].sort_values(by=['scraped_at', 'sentiment_score'],
                                                                                   ascending=[False, False])

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        with st.container(border=True):
            st.markdown("### Berita Negatif")
            if not neg_news.empty:
                with st.container(height=380):
                    for _, row in neg_news.iterrows():
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 8px; border-left: 5px solid #EF553B; background-color: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.1); margin-bottom: 10px;">
                            <h4 style="margin:0; font-size: 15px;"><a href="{row['url']}" target="_blank" style="text-decoration:none; color: #4DA6FF;">{row['title']}</a></h4>
                            <div style="margin-top: 5px; font-size: 12px; opacity: 0.7;"><b>{row['date_str']}</b> | {row['source']}</div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.success("Aman. Tidak ada berita negatif.")

    with col_n2:
        with st.container(border=True):
            st.markdown("### Tweet Negatif")
            if not neg_tweets.empty:
                with st.container(height=380):
                    for _, row in neg_tweets.iterrows():
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 8px; border-left: 5px solid #EF553B; background-color: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.1); margin-bottom: 10px;">
                            <div style="font-size: 14px; margin-bottom: 5px;">"{row['clean_text']}"</div>
                            <div style="font-size: 12px; opacity: 0.7;"><b>@{row['username']}</b> | {row['date_str']}</div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.success("Aman. Tidak ada tweet negatif.")


# ==========================================
# HALAMAN 2: ANALISIS BERITA
# ==========================================
def show_news_analytics(df_news, ai_cache):
    st.title("Analisis Media & Berita")

    if df_news.empty:
        st.warning("Belum ada data berita yang diproses.")
        return

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Berita", len(df_news), delta=" ", delta_color="off")
        c2.metric("Sentimen Positif", len(df_news[df_news['sentiment_label'] == 'Positive']), delta="Good")
        c3.metric("Sentimen Negatif", len(df_news[df_news['sentiment_label'] == 'Negative']), delta="-Alert",
                  delta_color="inverse")
        c4.metric("Sentimen Netral", len(df_news[df_news['sentiment_label'] == 'Neutral']), delta=" ",
                  delta_color="off")

    st.divider()

    sel_sources, sel_sentiments = [], []

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            st.subheader("Tren Publikasi Berita Harian")
            time_opt_news = st.radio("Rentang Waktu", ["1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun", "Semua Waktu"],
                                     index=0, horizontal=True, label_visibility="collapsed", key="time_news")

            df_news_trend = filter_by_time(df_news, 'final_date', time_opt_news)
            trend_news = df_news_trend.groupby(['date', 'sentiment_label']).size().reset_index(name='jumlah')
            fig_trend = px.line(trend_news, x='date', y='jumlah', color='sentiment_label',
                                color_discrete_map=get_color_map(), markers=True)
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Top 10 Media")
            top_sources_list = df_news['source'].value_counts().head(10).index.tolist()
            df_top_source = df_news[df_news['source'].isin(top_sources_list)]
            source_sentiment = df_top_source.groupby(['source', 'sentiment_label']).size().reset_index(name='jumlah')
            fig_source = px.bar(source_sentiment, x='jumlah', y='source', color='sentiment_label', orientation='h',
                                color_discrete_map=get_color_map(), barmode='stack')
            fig_source.update_yaxes(categoryorder='array', categoryarray=top_sources_list[::-1])
            fig_source.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

            try:
                bar_event = st.plotly_chart(fig_source, use_container_width=True, on_select="rerun", key="bar_news")
                if bar_event and bar_event.get("selection") and bar_event["selection"]["points"]:
                    sel_sources = [p["y"] for p in bar_event["selection"]["points"]]
            except TypeError:
                st.plotly_chart(fig_source, use_container_width=True)

    col3, col4 = st.columns([2, 1])
    with col3:
        with st.container(border=True):
            st.subheader("Word Cloud Judul Berita")
            media_words = set(re.findall(r'\b\w+\b', " ".join(df_news['source'].dropna().astype(str).unique()).lower()))
            wc_buffer = plot_wordcloud(df_news['title'].dropna().tolist(), colormap='magma',
                                       dynamic_stopwords=media_words)
            if wc_buffer: st.markdown(render_image_html(wc_buffer), unsafe_allow_html=True)
    with col4:
        with st.container(border=True):
            st.subheader("Proporsi Sentimen")
            sentiment_counts = df_news['sentiment_label'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentimen', 'Jumlah']
            fig_pie = px.pie(sentiment_counts, names='Sentimen', values='Jumlah', hole=0.4, color='Sentimen',
                             color_discrete_map=get_color_map())
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

            try:
                pie_event = st.plotly_chart(fig_pie, use_container_width=True, on_select="rerun", key="pie_news")
                if pie_event and pie_event.get("selection") and pie_event["selection"]["points"]:
                    sel_sentiments = [p["label"] for p in pie_event["selection"]["points"]]
            except TypeError:
                st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    with st.spinner("Mengambil analisis berita..."):
        if ai_cache is not None and ai_cache.get('summary_news'):
            render_tldr_ui(ai_cache['summary_news'], title="Ringkasan Analisis Eksekutif (Berita)")
        else:
            st.info("AI belum selesai merangkum berita.")

    with st.container(border=True):
        st.subheader("Tabel Data Mentah")

        cf1, cf2, cf3 = st.columns(3)
        cari_berita = cf1.text_input("Cari Judul...", key="cari_berita")

        filter_sentimen = cf2.multiselect("Filter Sentimen", options=df_news['sentiment_label'].unique(),
                                          default=sel_sentiments)
        filter_sumber = cf3.multiselect("Filter Media", options=df_news['source'].unique(), default=sel_sources)

        df_tampil = df_news.copy()
        if cari_berita:
            df_tampil = df_tampil[df_tampil['title'].str.contains(cari_berita, case=False, na=False)]
        if filter_sentimen:
            df_tampil = df_tampil[df_tampil['sentiment_label'].isin(filter_sentimen)]
        if filter_sumber:
            df_tampil = df_tampil[df_tampil['source'].isin(filter_sumber)]

        df_show = df_tampil.sort_values(by='date', ascending=False)[
            ['date_str', 'source', 'title', 'sentiment_label', 'sentiment_score', 'url']]

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "url": st.column_config.LinkColumn("Link Berita", display_text="Buka Artikel")
            }
        )


# ==========================================
# HALAMAN 3: ANALISIS TWITTER (X)
# ==========================================
def show_twitter_analytics(df_tweets, ai_cache):
    st.title("Analisis Cuitan X (Twitter)")
    if df_tweets.empty:
        st.warning("Belum ada data cuitan yang diproses.")
        return

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cuitan", len(df_tweets), delta=" ", delta_color="off")
        c2.metric("Sentimen Positif", len(df_tweets[df_tweets['sentiment_label'] == 'Positive']), delta="Good")
        c3.metric("Sentimen Negatif", len(df_tweets[df_tweets['sentiment_label'] == 'Negative']), delta="-Alert",
                  delta_color="inverse")
        c4.metric("Sentimen Netral", len(df_tweets[df_tweets['sentiment_label'] == 'Neutral']), delta=" ",
                  delta_color="off")

    st.divider()

    sel_users, sel_sentiments = [], []

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            st.subheader("Tren Obrolan Harian")
            time_opt_tw = st.radio("Rentang Waktu", ["1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun", "Semua Waktu"],
                                   index=0, horizontal=True, label_visibility="collapsed", key="time_tw")

            df_tw_trend = filter_by_time(df_tweets, 'scraped_at', time_opt_tw)
            trend_tw = df_tw_trend.groupby(['date', 'sentiment_label']).size().reset_index(name='jumlah')
            fig_trend = px.line(trend_tw, x='date', y='jumlah', color='sentiment_label',
                                color_discrete_map=get_color_map(),
                                markers=True)
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Top 10 Akun Paling Aktif")
            top_users_list = df_tweets['username'].value_counts().head(10).index.tolist()
            df_top_users = df_tweets[df_tweets['username'].isin(top_users_list)]
            user_sentiment = df_top_users.groupby(['username', 'sentiment_label']).size().reset_index(name='jumlah')
            fig_user = px.bar(user_sentiment, x='jumlah', y='username', color='sentiment_label', orientation='h',
                              color_discrete_map=get_color_map(), barmode='stack')
            fig_user.update_yaxes(categoryorder='array', categoryarray=top_users_list[::-1])
            fig_user.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

            try:
                bar_event = st.plotly_chart(fig_user, use_container_width=True, on_select="rerun", key="bar_tw")
                if bar_event and bar_event.get("selection") and bar_event["selection"]["points"]:
                    sel_users = [p["y"] for p in bar_event["selection"]["points"]]
            except TypeError:
                st.plotly_chart(fig_user, use_container_width=True)

    col3, col4 = st.columns([2, 1])
    with col3:
        with st.container(border=True):
            st.subheader("Word Cloud Cuitan")
            user_words = set(
                re.findall(r'\b\w+\b', " ".join(df_tweets['username'].dropna().astype(str).unique()).lower()))
            wc_buffer = plot_wordcloud(df_tweets['clean_text'].dropna().tolist(), colormap='Blues',
                                       dynamic_stopwords=user_words)
            if wc_buffer: st.markdown(render_image_html(wc_buffer), unsafe_allow_html=True)
    with col4:
        with st.container(border=True):
            st.subheader("Proporsi Sentimen")
            sentiment_counts = df_tweets['sentiment_label'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentimen', 'Jumlah']
            fig_pie = px.pie(sentiment_counts, names='Sentimen', values='Jumlah', hole=0.4, color='Sentimen',
                             color_discrete_map=get_color_map())
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

            try:
                pie_event = st.plotly_chart(fig_pie, use_container_width=True, on_select="rerun", key="pie_tw")
                if pie_event and pie_event.get("selection") and pie_event["selection"]["points"]:
                    sel_sentiments = [p["label"] for p in pie_event["selection"]["points"]]
            except TypeError:
                st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    with st.spinner("Mengambil analisis Twitter..."):
        if ai_cache is not None and ai_cache.get('summary_tweets'):
            render_tldr_ui(ai_cache['summary_tweets'], title="Ringkasan Analisis Eksekutif (Twitter)")
        else:
            st.info("AI belum selesai merangkum Twitter.")

    with st.container(border=True):
        st.subheader("Tabel Data Mentah")

        cf1, cf2, cf3 = st.columns([2, 1, 1])
        cari_cuitan = cf1.text_input("Cari Cuitan / Username...", key="cari_cuitan")
        filter_sentimen = cf2.multiselect("Filter Sentimen", options=df_tweets['sentiment_label'].unique(),
                                          default=sel_sentiments, key="sentimen_tw")
        filter_sumber = cf3.multiselect("Filter Akun", options=df_tweets['username'].unique(), default=sel_users,
                                        key="akun_tw")

        df_tampil = df_tweets.copy()
        if cari_cuitan:
            df_tampil = df_tampil[
                df_tampil['clean_text'].str.contains(cari_cuitan, case=False, na=False) |
                df_tampil['username'].str.contains(cari_cuitan, case=False, na=False)
                ]
        if filter_sentimen:
            df_tampil = df_tampil[df_tampil['sentiment_label'].isin(filter_sentimen)]
        if filter_sumber:
            df_tampil = df_tampil[df_tampil['username'].isin(filter_sumber)]

        df_show_tw = df_tampil.sort_values(by='date', ascending=False)[
            ['date_str', 'username', 'clean_text', 'sentiment_label', 'sentiment_score', 'tweet_url']]

        st.dataframe(
            df_show_tw,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "tweet_url": st.column_config.LinkColumn("Link Cuitan", display_text="Buka Cuitan")
            }
        )


# ==========================================
# HALAMAN 4: PENGATURAN
# ==========================================
def show_settings():
    st.title("Pengaturan")
    engine = get_db_engine()

    with st.container(border=True):
        st.subheader("Tambah Kata Kunci Baru")
        c1, c2 = st.columns([3, 1])
        new_kw = c1.text_input("Kata Kunci", label_visibility="collapsed", placeholder="Ketik keyword baru...")

        if c2.button("Simpan", type="primary", use_container_width=True):
            current_time = time.time()
            if current_time - st.session_state['last_action_time'] < 2.0:
                st.toast("Terlalu banyak permintaan! Tunggu 2 detik.")
            else:
                st.session_state['last_action_time'] = current_time
                is_safe, msg = validate_security(new_kw)

                if not is_safe:
                    st.toast(f"{msg}")
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
                        st.toast("Gagal menyimpan data!")

                    if is_success:
                        st.toast(f"Berhasil menambahkan keyword: {new_kw}")
                        time.sleep(1.2)
                        st.cache_data.clear()
                        st.rerun()
                    elif is_duplicate:
                        st.toast(f"Keyword '{new_kw}' sudah ada!")

    with st.container(border=True):
        st.subheader("Daftar Keyword Aktif")
        try:
            df_kw = pd.read_sql("SELECT * FROM keywords ORDER BY id DESC", engine)
            for _, row in df_kw.iterrows():
                col_a, col_b = st.columns([4, 1])
                col_a.text(f"• {row['keyword']}")
                if col_b.button("Hapus", key=f"del_{row['id']}", use_container_width=True):
                    try:
                        with engine.begin() as conn:
                            conn.execute(text("DELETE FROM keywords WHERE id=:id"), {"id": row['id']})
                        st.toast(f"Keyword '{row['keyword']}' berhasil dihapus!")
                        time.sleep(1.2)
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.toast("Gagal menghapus data!")
        except Exception as e:
            st.error(f"Gagal memuat daftar keyword: {e}")


# ==========================================
# UI FLOATING CHATBOT
# ==========================================
def show_floating_chat():
    if 'chat_expanded' not in st.session_state: st.session_state['chat_expanded'] = False
    chat_w, chat_h, cont_h = ("800px", "85vh", 600) if st.session_state['chat_expanded'] else ("420px", "600px", 400)

    st.markdown(f"""
        <style>
        div[data-testid="stPopover"] {{ position: fixed !important; bottom: 30px !important; right: 40px !important; z-index: 999999 !important; width: auto !important; }}
        div[data-testid="stPopover"] > button {{ width: 65px !important; height: 65px !important; border-radius: 50% !important; background-color: #00CC96 !important; color: white !important; border: none !important; padding: 0 !important; display: flex !important; justify-content: center !important; align-items: center !important; box-shadow: 0 6px 16px rgba(0,0,0,0.4) !important; transition: transform 0.3s ease; }}
        div[data-testid="stPopover"] > button p {{ font-size: 32px !important; margin: 0 !important; }}
        div[data-testid="stPopover"] > button:hover {{ transform: scale(1.1); }}
        div[data-testid="stPopoverBody"] {{ width: {chat_w} !important; height: {chat_h} !important; max-width: 95vw !important; max-height: 95vh !important; padding: 20px !important; border-radius: 15px !important; border: 1px solid #555 !important; resize: both !important; overflow: auto !important; transition: width 0.3s ease, height 0.3s ease; }}
        </style>""", unsafe_allow_html=True)

    with st.popover("💬 Chatbot"):
        c1, c2 = st.columns([4, 1])
        c1.markdown("###"
                    "Asisten AI")
        if c2.button("⛶" if not st.session_state['chat_expanded'] else "🗕"):
            st.session_state['chat_expanded'] = not st.session_state['chat_expanded']
            st.rerun()

        st.divider()
        with st.container(height=cont_h):
            if not st.session_state['chat_history']: st.markdown(
                "<div style='text-align:center; color:#888; margin-top:50px;'><i>Ketik pertanyaan di bawah.</i></div>",
                unsafe_allow_html=True)
            for msg in st.session_state['chat_history']:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

            if st.session_state['chat_history'] and st.session_state['chat_history'][-1]['role'] == 'user':
                with st.chat_message("assistant"):
                    with st.spinner("Menganalisis..."):
                        try:
                            # --- 1. TARIK KEYWORD AKTIF DARI DATABASE ---
                            engine = get_db_engine()
                            df_kw = pd.read_sql("SELECT keyword FROM keywords", engine)
                            # Gabungkan semua keyword aktif (misal: "UNSIKA, Rektor UNSIKA")
                            keyword_aktif = ", ".join(df_kw['keyword'].tolist()) if not df_kw.empty else "Institusi"

                            # --- 2. LEMPAR KEYWORD KE FUNGSI AI ---
                            history_untuk_konteks = st.session_state['chat_history'][:-1]
                            answer = ask_bot(
                                query=st.session_state['chat_history'][-1]['content'],
                                chat_history=history_untuk_konteks,
                                keyword=keyword_aktif  # <--- INI PARAMETER BARUNYA
                            )

                            st.session_state['chat_history'].append({"role": "assistant", "content": answer})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal memanggil AI: {e}")

        if prompt := st.chat_input("Tanyakan isu reputasi kampus..."):
            cur_time = time.time()
            if cur_time - st.session_state['last_action_time'] < 3.0:
                st.toast("Tunggu 3 detik.")
            else:
                st.session_state['last_action_time'] = cur_time
                safe, msg = validate_security(prompt)
                if not safe:
                    st.toast(msg)
                    st.session_state['chat_history'].extend([{"role": "user", "content": prompt}, {"role": "assistant",
                                                                                                   "content": f"Akses ditolak. {msg}"}])
                else:
                    st.session_state['chat_history'].append({"role": "user", "content": prompt})
                st.rerun()


# ==========================================
# FUNGSI MAIN
# ==========================================
def main():
    df_news, df_tweets = load_all_data()
    ai_cache = load_ai_cache()

    with st.sidebar:
        st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>Reputation Monitoring System</h1>",
                    unsafe_allow_html=True)

        selected = option_menu(None, ["Utama", "Berita (Media)", "X (Cuitan)", "Pengaturan"],
                               icons=["house", "newspaper", "twitter", "gear"], default_index=0,
                               styles={
                                   "container": {"padding": "0!important", "background-color": "transparent"},
                                   "icon": {"font-size": "16px"},
                                   "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px"},
                                   "nav-link-selected": {"background-color": "#00CC96", "color": "white"}
                               })
        st.markdown("---")
        st.caption("Diperbarui otomatis via Airflow.")

    if selected == "Utama":
        show_overview(df_news, df_tweets, ai_cache)
    elif selected == "Berita (Media)":
        show_news_analytics(df_news, ai_cache)
    elif selected == "X (Cuitan)":
        show_twitter_analytics(df_tweets, ai_cache)
    elif selected == "Pengaturan":
        show_settings()

    show_floating_chat()


if __name__ == "__main__":
    main()