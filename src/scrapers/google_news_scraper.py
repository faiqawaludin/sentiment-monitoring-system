import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import urllib.parse
import sys
import os
from sqlalchemy import text

# --- SETUP PATH ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from src.utils.db import get_db_engine
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.db import get_db_engine


class GoogleNewsScraper:
    def __init__(self, language='id-ID', location='ID'):
        self.base_url = "https://news.google.com/rss/search"
        self.params = {
            'hl': language,
            'gl': location,
            'ceid': f'{location}:{language.split("-")[0]}'
        }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def scrape(self, query, max_items=100):
        encoded_query = urllib.parse.quote(query)
        url = f"{self.base_url}?q={encoded_query}"

        try:
            response = requests.get(url, params=self.params, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, features='xml')
            items = soup.find_all('item')

            news_data = []
            for item in items:
                if len(news_data) >= max_items:
                    break

                title = item.title.text if item.title else "No Title"
                link = item.link.text if item.link else "No Link"
                # Mengambil text tanggal publikasi dari RSS
                pub_date = item.pubDate.text if item.pubDate else str(datetime.now())
                source = item.source.text if item.source else "Google News"

                news_data.append({
                    'scraped_at': datetime.now(),
                    'keyword': query,
                    'title': title,
                    'source': source,
                    'url': link,
                    'published_date': pub_date
                })

            return pd.DataFrame(news_data)

        except Exception as e:
            print(f"   ❌ Error scraping '{query}': {e}")
            return pd.DataFrame()


def run_full_scraping_job():
    print("🚀 Memulai Job Scraping Google News (Mode Dinamis via Database)...")

    scraper = GoogleNewsScraper()
    engine = get_db_engine()

    # --- 1. AMBIL KEYWORDS DARI DB ---
    try:
        # Ambil keyword yang source-nya 'news' atau 'all'
        query_keys = "SELECT keyword FROM keywords WHERE source IN ('news', 'all')"
        df_keys = pd.read_sql(query_keys, engine)

        if not df_keys.empty:
            KEYWORDS = df_keys['keyword'].tolist()
            print(f"📋 Menggunakan {len(KEYWORDS)} Kata Kunci dari Database: {KEYWORDS}")
        else:
            print("⚠️ Tabel keywords kosong. Menggunakan default.")
            KEYWORDS = ["UNSIKA", "Universitas Singaperbangsa Karawang"]
    except Exception as e:
        print(f"⚠️ Gagal membaca keywords (Error: {e}). Menggunakan default.")
        KEYWORDS = ["UNSIKA", "Universitas Singaperbangsa Karawang"]

    # --- 2. CEK URL EXISTING (Anti Duplikat) ---
    try:
        existing_urls = pd.read_sql("SELECT url FROM news_raw", engine)['url'].tolist()
        print(f"📚 Database memiliki {len(existing_urls)} berita lama.")
    except Exception as e:
        existing_urls = []
        print("📚 Database kosong atau tabel belum ada.")

    total_saved = 0

    # --- 3. LOOPING SCRAPING ---
    for keyword in KEYWORDS:
        print(f"🔎 Sedang mencari: '{keyword}'...")

        try:
            df = scraper.scrape(keyword, max_items=50)

            if not df.empty:
                # FILTER: Hanya ambil yang URL-nya BELUM ADA di database
                df_new = df[~df['url'].isin(existing_urls)]

                if not df_new.empty:
                    try:
                        df_new.to_sql('news_raw', engine, if_exists='append', index=False)
                        print(f"   ✅ Menyimpan {len(df_new)} berita BARU.")

                        # Update list lokal biar duplikat antar-keyword juga terfilter
                        existing_urls.extend(df_new['url'].tolist())
                        total_saved += len(df_new)
                    except Exception as e:
                        print(f"   ⚠️ Gagal menyimpan ke DB: {e}")
                else:
                    print("   ℹ️  Dapat berita, tapi semuanya duplikat (Skip).")
            else:
                print("   ⚠️ Tidak ditemukan berita di Google.")

        except Exception as e:
            print(f"   ❌ Error sistem pada keyword '{keyword}': {e}")

    print(f"🏁 Job Selesai. Total {total_saved} berita baru ditambahkan hari ini.")


if __name__ == "__main__":
    run_full_scraping_job()