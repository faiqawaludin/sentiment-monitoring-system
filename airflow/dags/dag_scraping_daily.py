from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# --- DEFINE WRAPPER FUNCTIONS (LAZY IMPORT) ---
# Trik: Import modul di dalam fungsi agar DAG tidak error saat loading awal

def scrape_news_wrapper():
    # 1. Daftarkan path src (Wajib)
    sys.path.append('/opt/airflow/src')
    # 2. Baru import modulnya di sini
    from scrapers.google_news_scraper import run_full_scraping_job
    # 3. Jalankan
    run_full_scraping_job()

def process_news_wrapper():
    sys.path.append('/opt/airflow/src')
    from processing.process_news import run_news_processing
    run_news_processing()

default_args = {
    'owner': 'remosy_admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
}

with DAG(
    'dag_scraping_daily',
    default_args=default_args,
    description='Scraping Berita & Analisis Sentimen Harian',
    schedule_interval='0 0,12 * * *', # Jam 00:00 dan 12:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['remosy', 'news', 'sentiment'],
) as dag:

    # Task 1: Ambil Berita
    task_scrape = PythonOperator(
        task_id='task_scrape_news',
        python_callable=scrape_news_wrapper # Panggil wrapper, bukan fungsi asli langsung
    )

    # Task 2: Analisis Sentimen
    task_process = PythonOperator(
        task_id='task_process_sentiment',
        python_callable=process_news_wrapper
    )

    # Urutan
    task_scrape >> task_process