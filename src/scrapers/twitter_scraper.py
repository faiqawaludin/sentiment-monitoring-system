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
        # ⚠️ GANTI DENGAN AUTH_TOKEN TERBARU DARI BROWSER ANDA!
        self.auth_token = "af2e1aedad90142b0929218e42fe61f75da1c0cb"

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
            # capture_output=True akan menangkap teks yang muncul di terminal
            process = subprocess.run(command, shell=True, capture_output=True, text=True)

            # ==========================================
            # 🚨 EARLY WARNING SYSTEM (DETEKSI TOKEN MATI)
            # ==========================================
            output_log = process.stdout.lower() + process.stderr.lower()

            # Ciri-ciri token mati di tweet-harvest biasanya memunculkan kata ini di terminal
            if "unauthorized" in output_log or "invalid token" in output_log or "rate limit" in output_log:
                print(f"⚠️ LOG ERROR TERDETEKSI:\n{output_log}")
                # KITA PAKSA CRASH AGAR AIRFLOW BERWARNA MERAH!
                raise ValueError(
                    "🚨 ALERT KRITIS: Auth Token Twitter kedaluwarsa atau diblokir! Segera perbarui di script.")
            # ==========================================

            expected_file = os.path.join("tweets-data", filename)

            if os.path.exists(expected_file):
                df = pd.read_csv(expected_file)

                # Jika file ada tapi isinya kosong
                if df.empty:
                    os.remove(expected_file)
                    return pd.DataFrame()

                clean_df = pd.DataFrame()
                clean_df['scraped_at'] = [datetime.now()] * len(df)
                clean_df['keyword'] = query

                # PERBAIKAN: Gunakan kolom 'username', bukan 'screen_name'
                clean_df['username'] = df.get('username', 'unknown')
                clean_df['full_text'] = df.get('full_text', '')

                # PERBAIKAN: Ambil langsung dari tweet-harvest, jangan dirakit manual
                clean_df['tweet_url'] = df.get('tweet_url', '')

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

        # Tambahan log untuk memantau hasil mentah dari scraper
        print(f"   -> Berhasil menarik {len(df)} tweet mentah dari X.")

        if not df.empty:
            # 2. FILTER DUPLIKAT
            df_new = df[~df['tweet_url'].isin(existing_links)]

            if not df_new.empty:
                try:
                    df_new.to_sql('tweets_raw', engine, if_exists='append', index=False)
                    print(f"   ✅ Menyimpan {len(df_new)} tweet BARU ke database.")

                    existing_links.extend(df_new['tweet_url'].tolist())
                    total_saved += len(df_new)
                except Exception as e:
                    print(f"   ❌ Gagal simpan: {e}")
            else:
                print("   ℹ️ Info: Tidak ada tweet baru (semua tweet sudah ada di database).")

    print(f"Job Twitter Selesai. Total {total_saved} tweet baru tersimpan.")


if __name__ == "__main__":
    run_twitter_scraping_job()