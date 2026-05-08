from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# --- WRAPPER LAZY IMPORT ---

def scrape_twitter_wrapper():
    sys.path.append('/opt/airflow/src')
    from scrapers.twitter_scraper import run_twitter_scraping_job
    run_twitter_scraping_job()

def process_twitter_wrapper():
    sys.path.append('/opt/airflow/src')
    # Perhatikan nama file importnya harus sesuai (process_tweets.py pakai 's')
    from processing.process_tweets import run_twitter_processing
    run_twitter_processing()

default_args = {
    'owner': 'remosy_admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 0,
}

with DAG(
    'dag_scraping_twitter',
    default_args=default_args,
    description='Scraping Twitter & Analisis Sentimen',
    schedule_interval='0 6,18 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['remosy', 'twitter', 'sentiment'],
) as dag:

    # Task 1: Scrape Twitter
    task_scrape = PythonOperator(
        task_id='task_scrape_twitter',
        python_callable=scrape_twitter_wrapper
    )

    # Task 2: Analisis Sentimen
    task_process = PythonOperator(
        task_id='task_process_sentiment',
        python_callable=process_twitter_wrapper
    )

    # Urutan
    task_scrape >> task_process