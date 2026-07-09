from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

# --- WRAPPERS FUNGSI PYTHON ---
def scrape_news():
    sys.path.append('/opt/airflow/src')
    from scrapers.google_news_scraper import run_full_scraping_job
    run_full_scraping_job()

def scrape_twitter():
    sys.path.append('/opt/airflow/src')
    from scrapers.twitter_scraper import run_twitter_scraping_job
    run_twitter_scraping_job()

def process_news():
    sys.path.append('/opt/airflow/src')
    from processing.process_news import run_news_processing
    run_news_processing()

def process_twitter():
    sys.path.append('/opt/airflow/src')
    from processing.process_tweets import run_twitter_processing
    run_twitter_processing()

def build_vector():
    sys.path.append('/opt/airflow/src')
    from chatbot.build_vector_db import build_faiss_index
    build_faiss_index()

def run_ai_summary():
    sys.path.append('/opt/airflow/src')
    from chatbot.ai_summarizer import run_ai_summarizer
    run_ai_summarizer()

# --- KONFIGURASI DAG ---
default_args = {
    'owner': 'remosy',
    'depends_on_past': False,
    'retries': 0,
}

with DAG(
    'remosy_master_end_to_end',
    default_args=default_args,
    description='Scrape Paralel -> Proses Sentimen -> Vector DB -> AI Summary',
    schedule_interval='0 0,12 * * *',  # Berjalan setiap jam 00:00 dan 12:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['remosy', 'end-to-end', 'master']
) as dag:

    # 1. Gerbong Ekstraksi (Berjalan Paralel)
    t_scrape_news = PythonOperator(task_id='scrape_news', python_callable=scrape_news)
    t_scrape_tw = PythonOperator(task_id='scrape_twitter', python_callable=scrape_twitter)

    # 2. Gerbong Transformasi (Membersihkan & Menilai Sentimen)
    t_process_news = PythonOperator(task_id='process_news', python_callable=process_news)
    t_process_tw = PythonOperator(task_id='process_twitter', python_callable=process_twitter)

    # 3. Gerbong Intelijen (Membangun Vector DB & Menghasilkan Ringkasan AI)
    t_vector = PythonOperator(task_id='update_vector_db', python_callable=build_vector)
    t_ai = PythonOperator(task_id='generate_ai_summary', python_callable=run_ai_summary)

    # --- ATURAN LALU LINTAS (DEPENDENCIES) ---
    t_scrape_news >> t_process_news
    t_scrape_tw >> t_process_tw

    # Vector DB dan AI Summary HANYA boleh berjalan JIKA kedua proses sentimen di atas sudah selesai
    [t_process_news, t_process_tw] >> t_vector >> t_ai