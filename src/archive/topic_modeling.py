import pandas as pd
import re
import os
import json
import time
from google import genai
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from dotenv import load_dotenv

# =========================================
# 0. SETUP & AUTH
# =========================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️ Gagal init client: {e}")


# =========================================
# 1. DATABASE
# =========================================
def get_db_engine():
    user = os.getenv("DB_USER", "remosy_user")
    password = os.getenv("DB_PASSWORD", "remosy_password")
    dbname = os.getenv("DB_NAME", "remosy_dw")

    if os.path.exists('/.dockerenv'):
        host = os.getenv("POSTGRES_HOST_DOCKER", "postgres")
        port = os.getenv("POSTGRES_PORT_DOCKER", "5432")
    else:
        host = os.getenv("POSTGRES_HOST_LOCAL", "localhost")
        port = os.getenv("POSTGRES_PORT_LOCAL", "5434")

    conn_str = f'postgresql://{user}:{password}@{host}:{port}/{dbname}'

    try:
        engine = create_engine(conn_str)
        return engine
    except Exception as e:
        print(f"❌ KONEKSI GAGAL: {e}")
        return create_engine(f'postgresql://remosy_user:remosy_password@localhost:5434/remosy_dw')


# =========================================
# 2. PREPROCESSING & STOPWORDS (UPGRADED)
# =========================================
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


def get_custom_stopwords():
    factory = StopWordRemoverFactory()
    sastrawi_stopwords = factory.get_stop_words()

    # Kumpulan kata tidak penting (Stopwords Ekstrem)
    custom_stopwords = [
        # 1. Kata Sambung/Tugas Dasar (Wajib hilang)
        'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'ini', 'itu',
        'juga', 'sudah', 'telah', 'ada', 'adalah', 'karena', 'oleh', 'saat', 'setelah', 'akan',
        'sedang', 'bagi', 'mari', 'yuk', 'hingga', 'lewat', 'sebagai', 'atau', 'serta',

        # 2. Kata Keterangan Waktu/Jumlah/Sifat Umum
        'baru', 'lama', 'besar', 'kecil', 'tinggi', 'rendah', 'baik', 'banyak', 'sedikit',
        'lebih', 'kurang', 'sangat', 'sekali', 'hari', 'tahun', 'bulan', 'minggu', 'jam',
        'pertama', 'kedua', 'ketiga', 'satu', 'dua', 'tiga',

        # 3. Kata Kerja Dasar (Terlalu umum)
        'bikin', 'buat', 'ucap', 'kata', 'ujar', 'sebut', 'milik', 'minta', 'ikut', 'terima',
        'kasih', 'buka', 'tutup', 'mulai', 'kerja', 'ajar', 'raih', 'kunjung', 'bangun',
        'lakukan', 'adakan', 'siap', 'jadi', 'dorong', 'gelar',

        # 4. Bahasa Gaul / Singkatan Twitter
        'yg', 'ga', 'gak', 'kalo', 'kl', 'dr', 'udah', 'sdh', 'bgt', 'aja', 'doang', 'tp', 'tapi',
        'sy', 'aku', 'gue', 'gw', 'ya', 'nih', 'tuh', 'mau', 'lagi', 'apa', 'knp', 'utk', 'sih', 'dong',

        # 5. Stopwords Spesifik Domain Kampus (Agar LDA mencari Isu-nya, bukan nama tempatnya)
        'unsika', 'universitas', 'singaperbangsa', 'karawang', 'kampus', 'mahasiswa', 'mahasiswi',
        'rektor', 'dosen', 'fakultas', 'prodi', 'jurusan', 'kelas', 'program', 'bem', 'ormawa', 'ukm',
        'prof', 'dr', 'univ', 'ptn', 'negeri', 'swasta',

        # 6. Kata Sambung Media (Noise dari Scraper)
        'inewsid', 'inews', 'id', 'tvberitacoid', 'tvberita', 'co', 'kompasianacom', 'kompasiana', 'com',
        'infokaid', 'infoka', 'wartakotalivecom', 'wartakotalive', 'beritapasundancom', 'karawangnewscom',
        'radarkarawang', 'metropolitan', 'pojoksatu', 'rakyat', 'jelata', 'tribunnews', 'sindonews',
        'kumparan', 'ayokarawang', 'detik', 'jpnn', 'medcom', 'kompas', 'liputan6', 'cnn', 'cnbc',
        'okezone', 'viva', 'suara', 'pikiran', 'merdeka', 'grid', 'bola', 'tempo', 'antara', 'media',
        'redaksi', 'halaman', 'baca', 'selengkapnya', 'sumber', 'foto', 'video', 'advertisement', 'editor',
        'wartawan', 'jawabarat', 'jabar', 'bekasi', 'purwakarta', 'indonesia', 'klik', 'disini', 'link'
    ]

    # Gabungkan, hapus duplikat
    return list(set(sastrawi_stopwords + custom_stopwords))


def clean_text_for_lda(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Hapus URL
    text = re.sub(r'(@\w+|#\w+)', '', text)  # Hapus Mention/Hashtag
    text = text.replace('.', ' ').replace('-', ' ').replace('/', ' ')
    text = re.sub(r'[^a-z\s]', '', text)  # Hapus angka & simbol
    text = re.sub(r'\s+', ' ', text).strip()

    # Stemming (Bisa dimatikan jika membuat proses LDA terlalu lambat)
    try:
        text = stemmer.stem(text)
    except:
        pass
    return text


# =========================================
# 3. FUNGSI UTAMA (AUTO-RETRY & FALLBACK)
# =========================================
def interpret_with_gemini(keywords_list):
    if not client: return []

    candidate_models = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-pro-latest"]
    prompt = f"""
    Kamu adalah analis media sosial. Simpulkan topik dari keywords ini.
    Keywords:
    {keywords_list}

    Output JSON Array Only (Tanpa Markdown):
    [
      {{
        "topic_id": 1,
        "title": "Judul Pendek",
        "summary": "Ringkasan satu kalimat.",
        "detail": "Buat 2 paragraf singkat berisi analisis mendalam tentang isu ini. Apa yang mungkin terjadi, pihak mana yang terlibat, dan bagaimana sentimen umumnya berdasarkan kata kunci tersebut."
      }}
    ]
    """

    for model_name in candidate_models:
        print(f"🤖 Mencoba Model: {model_name} ...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                if response.text:
                    print(f"✅ Sukses dengan {model_name}!")
                    clean_json = response.text.replace('```json', '').replace('```', '').strip()
                    return json.loads(clean_json)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    time.sleep(60)
                else:
                    break
    return []

def get_lda_topics(n_topics=5):
    engine = get_db_engine()

    # --- 🚨 SOLUSI TOPIK BASI: Filter data maksimal 30 hari ke belakang ---
    q_tweets = "SELECT full_text as text_content FROM tweets_raw WHERE scraped_at >= NOW() - INTERVAL '30 days' ORDER BY scraped_at DESC LIMIT 1000"
    q_news = "SELECT clean_text as text_content FROM news_processed WHERE processed_at >= NOW() - INTERVAL '30 days' ORDER BY id DESC LIMIT 500"

    try:
        df_tweets = pd.read_sql(q_tweets, engine)
        df_news = pd.read_sql(q_news, engine)
        df_all = pd.concat([df_tweets, df_news], ignore_index=True)
    except Exception as e:
        return {"error": str(e)}

    if df_all.empty: return {}

    df_all['clean'] = df_all['text_content'].apply(clean_text_for_lda)
    stopwords = get_custom_stopwords()

    try:
        tf_vectorizer = CountVectorizer(max_df=0.85, min_df=3, stop_words=stopwords, token_pattern=r'\b[a-zA-Z]{3,}\b')
        dtm = tf_vectorizer.fit_transform(df_all['clean'])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
    except ValueError as ve:
        return {}

    feature_names = tf_vectorizer.get_feature_names_out()
    raw_topics_str = ""
    raw_data_for_ui = {}

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_indices]
        raw_topics_str += f"Topik {topic_idx + 1}: {', '.join(top_words)}\n"
        raw_data_for_ui[topic_idx + 1] = top_words

    gemini_results = interpret_with_gemini(raw_topics_str)

    final_results = []
    if gemini_results:
        for item in gemini_results:
            t_id = item.get('topic_id')
            keywords = raw_data_for_ui.get(t_id, [])
            final_results.append({
                "id": t_id,
                "title": item.get('title', f"Topik {t_id}"),
                "summary": item.get('summary', "Tidak ada ringkasan."),
                "detail": item.get('detail', "Penjelasan rinci tidak tersedia."), # Tangkap data detail
                "keywords": keywords
            })
    return final_results

def get_lda_topics(n_topics=5):
    engine = get_db_engine()

    q_tweets = "SELECT full_text as text_content FROM tweets_raw ORDER BY scraped_at DESC LIMIT 1000"
    q_news = "SELECT clean_text as text_content FROM news_processed ORDER BY id DESC LIMIT 500"

    try:
        df_tweets = pd.read_sql(q_tweets, engine)
        df_news = pd.read_sql(q_news, engine)
        df_all = pd.concat([df_tweets, df_news], ignore_index=True)
    except Exception as e:
        return {"error": str(e)}

    if df_all.empty: return {}

    # 1. Bersihkan Teks
    df_all['clean'] = df_all['text_content'].apply(clean_text_for_lda)
    stopwords = get_custom_stopwords()

    try:
        # 2. Vektorisasi Teks (Dengan aturan ketat)
        # token_pattern=r'\b[a-zA-Z]{3,}\b' memaksa hanya kata minimal 3 huruf yang diambil (mengusir 'yg', 'di', 'ke')
        tf_vectorizer = CountVectorizer(
            max_df=0.85,  # Abaikan kata yang muncul di lebih dari 85% dokumen
            min_df=3,  # Kata harus muncul minimal di 3 dokumen berbeda
            stop_words=stopwords,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        dtm = tf_vectorizer.fit_transform(df_all['clean'])

        # 3. Proses LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
    except ValueError as ve:
        print(f"⚠️ ValueError LDA: {ve}")
        return {}

    feature_names = tf_vectorizer.get_feature_names_out()
    raw_topics_str = ""
    raw_data_for_ui = {}

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_indices]
        raw_topics_str += f"Topik {topic_idx + 1}: {', '.join(top_words)}\n"
        raw_data_for_ui[topic_idx + 1] = top_words

    gemini_results = interpret_with_gemini(raw_topics_str)

    final_results = []
    if gemini_results:
        for item in gemini_results:
            t_id = item.get('topic_id')
            keywords = raw_data_for_ui.get(t_id, [])
            final_results.append({
                "id": t_id,
                "title": item.get('title', f"Topik {t_id}"),
                "summary": item.get('summary', "Tidak ada ringkasan."),
                "keywords": keywords
            })
    else:
        for t_id, words in raw_data_for_ui.items():
            final_results.append({
                "id": t_id,
                "title": f"Topik {t_id} (Manual)",
                "summary": "Analisis AI tidak tersedia (Kuota Habis).",
                "keywords": words
            })

    return final_results