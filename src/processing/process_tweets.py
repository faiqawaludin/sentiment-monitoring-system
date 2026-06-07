import sys
import os
import pandas as pd
from sqlalchemy import text
from datetime import datetime

# Tambahkan path root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.db import get_db_engine
from src.processing.text_cleaner import clean_text
from src.processing.sentiment_indobert import predict_sentiment


# --- 🧠 LOGIKA CERDAS: FIX SENTIMEN (REPUTATION-AWARE) ---
def refine_sentiment(text, original_label, original_score):
    """
    Memperbaiki kesalahan AI karena IndoBERT melihat kalimat sebagai fakta objektif.
    """
    text_lower = text.lower()
    safe_label = str(original_label).capitalize()

    # 🚨 1. KASUS FATAL (Reputasi Kampus Rusak) - DICEK PERTAMA KALI
    fatal_negatives = ['demo', 'protes', 'korupsi', 'pelecehan', 'pungli', 'pidana', 'tersangka', 'mangkrak', 'janggal', 'kejanggalan']
    if any(w in text_lower for w in fatal_negatives):
        return 'Negative', 0.95

    # ♻️ 2. KASUS SAMPAH & LINGKUNGAN
    if 'sampah' in text_lower or 'limbah' in text_lower:
        positive_context = ['pengelolaan', 'mengelola', 'kelola', 'bank sampah', 'daur ulang', 'inovasi', 'bersih', 'solusi']
        if any(word in text_lower for word in positive_context):
            return 'Positive', 0.95

    # 🛣️ 3. KASUS JALAN & BENCANA
    if 'jalan' in text_lower and any(w in text_lower for w in ['diperbaiki', 'mulus', 'rampung']):
        return 'Positive', 0.90
    if any(w in text_lower for w in ['banjir', 'bencana', 'kecelakaan']):
        solution_context = ['mitigasi', 'bantuan', 'membantu', 'donasi', 'evakuasi', 'selamat', 'inovasi', 'teknologi', 'cegah']
        if any(word in text_lower for word in solution_context):
            return 'Positive', 0.95

        # 🎓 4. KEGIATAN AKADEMIK, PENGABDIAN & PRESTASI (BARU!)
        academic_positive = [
            'seminar', 'workshop', 'kkn', 'pelatihan', 'gandeng', 'kerja sama', 'kerjasama',
            'kuliah umum', 'lokakarya', 'penelitian', 'inovasi', 'prestasi', 'juara', 'lulus',
            'wisuda', 'hibah', 'pendanaan', 'pemeriksaan', 'sosialisasi', 'pengabdian', 'bantuan',
            'beasiswa', 'penghargaan', 'unggulan', 'solusi', 'terjunkan', 'bantu', 'bina', 'membangun',
            'percepat', 'maju', 'sukses', 'kontribusi', 'mendukung', 'kolaborasi', 'inovatif', 'kreatif',
            'peduli', 'berbagi', 'santunan', 'kampanye', 'dorong', 'gelar',
            'akreditasi', 'unggul', 'terakreditasi'  # <-- INI TAMBAHANNYA
        ]
    if any(word in text_lower for word in academic_positive):
        return 'Positive', 0.90

    # 📉 5. PENDETEKSI REPUTASI NEGATIF LAINNYA
    if safe_label == 'Neutral' or safe_label == 'Positive':
        neg_keywords = [
            'dugaan', 'polemik', 'ultimatum', 'gagal', 'kecewa', 'batal', 'rugi',
            'sanksi', 'pelanggaran', 'tuntut', 'kecam', 'soroti', 'kontroversi', 'darurat'
        ]
        if any(word in text_lower for word in neg_keywords):
            return 'Negative', 0.85

    # Kalau tidak masuk pengecualian apapun, kembalikan label aslinya
    return safe_label, original_score


def run_twitter_processing():
    print("🚀 Memulai Job Pemrosesan Twitter (Cleaning + Sentiment + Relevance Filter)...")

    engine = get_db_engine()

    # 1. Ambil tweet yang BELUM diproses
    query = """
            SELECT t.id, t.full_text
            FROM tweets_raw t
                     LEFT JOIN tweets_processed p ON t.id = p.tweet_id
            WHERE p.tweet_id IS NULL LIMIT 500
            """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Error membaca database: {e}")
        return

    if df.empty:
        print("✅ Tidak ada tweet baru untuk diproses.")
        return

    print(f"📦 Ditemukan {len(df)} tweet baru. Sedang memproses...")

    results = []

    for index, row in df.iterrows():
        tweet_id = row['id']
        tweet_asli = row['full_text']

        # --- 🚨 FILTER RELEVANSI (SATPAM KAMPUS UNTUK TWITTER) ---
        # Cek relevansi di teks aslinya (tweet_asli) sebelum di-clean
        tweet_lower = str(tweet_asli).lower()
        if not ('unsika' in tweet_lower or 'singaperbangsa' in tweet_lower):
            try:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM tweets_raw WHERE id = :id"), {"id": tweet_id})
                print(f"   🗑️ HAPUS (Tweet Nyasar): '{tweet_asli[:45]}...'")
            except Exception as e:
                print(f"   ⚠️ Gagal menghapus tweet nyasar: {e}")

            # Langsung skip, jangan buang-buang resource AI
            continue
        # --------------------------------------------

        # A. Cleaning (Otomatis hapus @username, link, RT)
        teks_bersih = clean_text(tweet_asli)

        # B. Prediksi AI Murni (IndoBERT)
        label, score = predict_sentiment(teks_bersih)

        # C. Terapkan Logika Pintar (Reputation-Aware)
        final_label, final_score = refine_sentiment(teks_bersih, label, score)

        # D. Tampung hasil akhir
        results.append({
            'tweet_id': tweet_id,
            'clean_text': teks_bersih,
            'sentiment_score': final_score,
            'sentiment_label': final_label,
            'processed_at': datetime.now()
        })

        # Log per 10 tweet biar ga nyepam
        if (index + 1) % 10 == 0:
            print(f"   [{index + 1}/{len(df)}] Processing...")

    # 3. Simpan DB
    if results:
        df_result = pd.DataFrame(results)
        try:
            df_result.to_sql('tweets_processed', engine, if_exists='append', index=False)
            print(f"✅ SUKSES! {len(df_result)} tweet berhasil dianalisis dengan otak hibrida baru.")
        except Exception as e:
            print(f"❌ Gagal menyimpan ke DB: {e}")


if __name__ == "__main__":
    run_twitter_processing()