import subprocess
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# --- SETUP PATH ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from src.utils.db import get_db_engine
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.db import get_db_engine


class TwitterScraper:
    def __init__(self):
        # Token Twitter Kamu
        self.auth_token = "18a41f13fc48eab001e0d9a4ab23d14b5311c2c9"

    def get_since_date(self, days=30):
        return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    def scrape(self, query, max_items=50):
        print(f"Mencari Tweet: {query}...")
        since_date = self.get_since_date(days=60)
        search_query = f"{query} since:{since_date}"
        filename = "temp_tweets.csv"

        command = (
            f"npx -y tweet-harvest@2.6.1 -o \"{filename}\" "
            f"-s \"{search_query}\" "
            f"--tab \"LATEST\" "
            f"-l {max_items} "
            f"--token {self.auth_token}"
        )

        try:
            process = subprocess.run(command, shell=True, capture_output=True, text=True)
            expected_file = os.path.join("tweets-data", filename)

            if os.path.exists(expected_file):
                df = pd.read_csv(expected_file)

                clean_df = pd.DataFrame()
                clean_df['scraped_at'] = [datetime.now()] * len(df)
                clean_df['keyword'] = query
                clean_df['username'] = df.get('screen_name', 'unknown')
                clean_df['full_text'] = df.get('full_text', '')
                clean_df['tweet_url'] = "https://twitter.com/" + clean_df['username'] + "/status/" + df.get('id_str',
                                                                                                            '').astype(
                    str)
                clean_df['likes'] = df.get('favorite_count', 0)
                clean_df['retweets'] = df.get('retweet_count', 0)

                os.remove(expected_file)
                return clean_df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"   Error Tweet-Harvest: {e}")
            return pd.DataFrame()


# --- FUNGSI UTAMA (DENGAN FILTER DUPLIKAT) ---
def run_twitter_scraping_job():
    print("Memulai Job Scraping Twitter (Mode Cerdas)...")

    KEYWORDS = ["unsika", "universitas singaperbangsa karawang", "mahasiswa unsika"]

    scraper = TwitterScraper()
    engine = get_db_engine()

    # 1. Cek Tweet yang sudah ada
    try:
        existing_links = pd.read_sql("SELECT tweet_url FROM tweets_raw", engine)['tweet_url'].tolist()
        print(f"Database saat ini memiliki {len(existing_links)} tweet.")
    except:
        existing_links = []

    total_saved = 0

    for keyword in KEYWORDS:
        df = scraper.scrape(keyword, max_items=50)

        if not df.empty:
            # 2. FILTER DUPLIKAT
            df_new = df[~df['tweet_url'].isin(existing_links)]

            if not df_new.empty:
                try:
                    df_new.to_sql('tweets_raw', engine, if_exists='append', index=False)
                    print(f"   Menyimpan {len(df_new)} tweet BARU.")

                    existing_links.extend(df_new['tweet_url'].tolist())
                    total_saved += len(df_new)
                except Exception as e:
                    print(f"   Gagal simpan: {e}")
            else:
                print("   info: Tidak ada tweet baru (semua duplikat).")

    print(f"Job Twitter Selesai. Total {total_saved} tweet baru tersimpan.")


if __name__ == "__main__":
    run_twitter_scraping_job()