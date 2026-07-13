import sys
import os
import pandas as pd
from datetime import datetime, date
from sqlalchemy import text
from dotenv import load_dotenv

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '../..'))
if root_project not in sys.path:
    sys.path.append(root_project)

# Load variabel dari .env
load_dotenv()

from src.utils.db import get_db_engine
from src.chatbot.llm_engine import generate_summary, generate_hot_topics

def run_ai_summarizer():
    print("🤖 Memulai Pre-komputasi Generative AI (Anti-Limit API)...")
    engine = get_db_engine()

    # ==========================================
    # 0. AMBIL KEYWORD DINAMIS
    # ==========================================
    # Dia akan nyari TARGET_INSTITUSI di .env, kalau nggak ada, otomatis pakai "UNSIKA"
    keyword_dinamis = os.getenv("TARGET_INSTITUSI", "UNSIKA")
    print(f"🔍 Menggunakan keyword institusi: {keyword_dinamis}")

    # 1. Buat Tabel Cache (Jika belum ada)
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ai_analysis_cache (
                    tanggal DATE PRIMARY KEY,
                    summary_news TEXT,
                    summary_tweets TEXT,
                    hot_topics TEXT,
                    updated_at TIMESTAMP
                )
            """))
    except Exception as e:
        print(f"❌ Gagal membuat tabel cache: {e}")
        return

    # 2. Tarik Data 90 Hari Terakhir
    print("📥 Menarik data teks dari database...")
    try:
        # Kita ambil 50 data terbaru agar tidak melebihi batas token LLM
        df_news = pd.read_sql("SELECT clean_text FROM news_processed WHERE processed_at >= NOW() - INTERVAL '90 days' ORDER BY processed_at DESC LIMIT 50", engine)
        df_tweets = pd.read_sql("SELECT clean_text FROM tweets_processed WHERE processed_at >= NOW() - INTERVAL '90 days' ORDER BY processed_at DESC LIMIT 50", engine)
    except Exception as e:
        print(f"❌ Gagal menarik data: {e}")
        return

    news_list = df_news['clean_text'].tolist() if not df_news.empty else []
    tweets_list = df_tweets['clean_text'].tolist() if not df_tweets.empty else []

    if not news_list and not tweets_list:
        print("⚠️ Tidak ada data untuk dirangkum hari ini.")
        return

    # 3. Panggil AI dengan KEYWORD DINAMIS
    print(f"🧠 Memanggil AI untuk Ringkasan Berita ({keyword_dinamis})...")
    summary_news = generate_summary(news_list, context_type="Berita Portal", keyword=keyword_dinamis)

    print(f"🧠 Memanggil AI untuk Ringkasan Cuitan ({keyword_dinamis})...")
    summary_tweets = generate_summary(tweets_list, context_type="Cuitan Twitter", keyword=keyword_dinamis)

    print(f"🔥 Memanggil AI untuk Topik Terhangat ({keyword_dinamis})...")
    # Penanganan aman: jika fungsi ini belum disetting untuk menerima keyword, dia tetap jalan.
    try:
        hot_topics = generate_hot_topics(news_list, tweets_list, keyword=keyword_dinamis)
    except TypeError:
        hot_topics = generate_hot_topics(news_list, tweets_list)

    # 4. Simpan ke Database (Upsert: Timpa jika tanggal hari ini sudah ada)
    print("💾 Menyimpan hasil AI ke PostgreSQL...")
    today_date = date.today()
    now_time = datetime.now()

    upsert_query = text("""
        INSERT INTO ai_analysis_cache (tanggal, summary_news, summary_tweets, hot_topics, updated_at)
        VALUES (:tgl, :s_news, :s_tweets, :ht, :upd)
        ON CONFLICT (tanggal)
        DO UPDATE SET
            summary_news = EXCLUDED.summary_news,
            summary_tweets = EXCLUDED.summary_tweets,
            hot_topics = EXCLUDED.hot_topics,
            updated_at = EXCLUDED.updated_at;
    """)

    try:
        with engine.begin() as conn:
            conn.execute(upsert_query, {
                "tgl": today_date,
                "s_news": summary_news,
                "s_tweets": summary_tweets,
                "ht": hot_topics,
                "upd": now_time
            })
        print(f"✅ SUKSES! Analisis AI untuk tanggal {today_date} berhasil disimpan secara permanen.")
    except Exception as e:
        print(f"❌ Gagal menyimpan ke DB: {e}")

if __name__ == "__main__":
    run_ai_summarizer()