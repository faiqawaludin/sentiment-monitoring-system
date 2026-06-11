import pandas as pd
from sqlalchemy import text
from textblob import TextBlob
import sys
import os
import re

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.db import get_db_engine


class SentimentProcessor:
    def __init__(self):
        self.engine = get_db_engine()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Hapus link, mention, hashtag, dan karakter aneh
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.lower().strip()

    # --- BAGIAN OTAK CERDAS (UPDATED) ---
    def get_smart_sentiment(self, text):
        text = text.lower()

        # 1. KAMUS POSITIF (Override)
        positive_context = [
            "olah sampah", "bank sampah", "biopori", "daur ulang",
            "inovasi", "prestasi", "juara", "menang", "penghargaan",
            "lolos", "beasiswa", "pengabdian", "kkn", "membangun",
            "edukasi", "sosialisasi", "hijau", "bersih", "bantu",
            "sukses", "resmi", "lulusan", "terbanyak", "akreditasi",
            "internasional", "unggul", "tepat guna", "mitigasi",
            "antisipasi", "cegah", "atasi", "solusi", "bantuan"
        ]

        # 2. KAMUS NEGATIF (DIPERLUAS & DIPERTAJAM)
        negative_context = [
            "korupsi", "demo", "ricuh", "bentrok", "sampah berserakan",
            "bau busuk", "banjir", "tenggelam", "macet parah",
            "pungli", "pelecehan", "lambat", "persulit",
            "keluh", "mengeluh", "tidak memadai", "kecewa",
            "gagal", "kegagalan", "kontroversi", "mangkrak",
            "ditunggangi", "kepentingan", "soroti", "kritik",
            "protes", "ancam", "menolak", "ditolak", "masalah",
            "dugaan", "pelanggaran", "sanksi", "tercoreng", "lamban"
        ]

        is_positive = any(word in text for word in positive_context)
        is_negative = any(word in text for word in negative_context)
        if is_negative and is_positive:
            return 'Positive', 0.6
        if is_negative:
            return 'Negative', -0.9
        if is_positive:
            return 'Positive', 0.9
        analysis = TextBlob(text)
        score = analysis.sentiment.polarity
        if score > 0.05:
            return 'Positive', score
        elif score < -0.05:
            return 'Negative', score
        else:
            return 'Neutral', 0
    def process_news(self):
        print("🔄 Memproses Sentimen Berita...")
        with self.engine.connect() as conn:
            # Ambil berita yang BELUM ada di tabel processed
            query = """
                    SELECT nr.id, nr.title
                    FROM news_raw nr
                             LEFT JOIN news_processed np ON nr.id = np.news_id
                    WHERE np.news_id IS NULL \
                    """
            df = pd.read_sql(query, conn)

        if df.empty:
            print("   ✅ Tidak ada berita baru untuk dinilai.")
            return

        print(f"   📊 Menilai {len(df)} berita baru...")
        results = []
        for _, row in df.iterrows():
            clean = self.clean_text(row['title'])
            label, score = self.get_smart_sentiment(clean)

            results.append({
                "news_id": row['id'],
                "clean_text": clean,
                "sentiment_label": label,
                "sentiment_score": score,
                "processed_at": pd.Timestamp.now()
            })

        if results:
            pd.DataFrame(results).to_sql('news_processed', self.engine, if_exists='append', index=False)
            print("   ✅ Selesai menyimpan sentimen berita.")

    def process_tweets(self):
        print("🔄 Memproses Sentimen Twitter...")
        with self.engine.connect() as conn:
            query = """
                    SELECT tr.id, tr.full_text
                    FROM tweets_raw tr
                             LEFT JOIN tweets_processed tp ON tr.id = tp.tweet_id
                    WHERE tp.tweet_id IS NULL \
                    """
            df = pd.read_sql(query, conn)

        if df.empty:
            print("   ✅ Tidak ada tweet baru untuk dinilai.")
            return

        print(f"   📊 Menilai {len(df)} tweet baru...")
        results = []
        for _, row in df.iterrows():
            clean = self.clean_text(row['full_text'])
            label, score = self.get_smart_sentiment(clean)

            results.append({
                "tweet_id": row['id'],
                "clean_text": clean,
                "sentiment_label": label,
                "sentiment_score": score,
                "processed_at": pd.Timestamp.now()
            })

        if results:
            pd.DataFrame(results).to_sql('tweets_processed', self.engine, if_exists='append', index=False)
            print("   ✅ Selesai menyimpan sentimen tweet.")


def run_sentiment_job():
    processor = SentimentProcessor()
    processor.process_news()
    processor.process_tweets()


if __name__ == "__main__":
    run_sentiment_job()