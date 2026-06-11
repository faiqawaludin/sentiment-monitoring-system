import sys
import os
import pandas as pd
from datetime import datetime, date
from sqlalchemy import text

# --- SETUP PATH ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.db import get_db_engine
from src.chatbot.llm_engine import generate_summary, generate_hot_topics

def run_ai_summarizer():
    print("🤖 Memulai Pre-komputasi Generative AI (Anti-Limit API)...")
    engine = get_db_engine()

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

    # 3. Panggil Gemini (Biarkan Airflow yang menunggu proses ini, bukan User!)
    print("🧠 Memanggil Gemini untuk Ringkasan Berita...")
    summary_news = generate_summary(news_list, context_type="Berita Portal")

    print("🧠 Memanggil Gemini untuk Ringkasan Cuitan...")
    summary_tweets = generate_summary(tweets_list, context_type="Cuitan Twitter")

    print("🔥 Memanggil Gemini untuk Topik Terhangat...")
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