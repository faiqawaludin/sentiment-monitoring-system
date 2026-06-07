import pandas as pd
from sqlalchemy import text
from transformers import pipeline
from datetime import datetime
import sys
import os

# --- SETUP PATH ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from src.utils.db import get_db_engine
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.db import get_db_engine


# --- FUNGSI PEMBERSIH TEKS SEDERHANA ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Hapus karakter aneh/simbol jika perlu
    return text.replace("\n", " ").strip()


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


# --- FUNGSI UTAMA ---
def run_news_processing():
    print("🚀 Memulai Job Pemrosesan Berita (Smart Sentiment & Relevance Filter)...")
    engine = get_db_engine()

    # 1. Ambil berita yang BELUM diproses
    # Logic: Ambil ID dari news_raw yang tidak ada di news_processed
    # 1. Ambil berita yang BELUM diproses menggunakan Left-Anti Join (Optimal)
    query = """
            SELECT nr.id, nr.title, nr.source
            FROM news_raw nr
                     LEFT JOIN news_processed np ON nr.id = np.news_id
            WHERE np.news_id IS NULL \
            """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"   ❌ Gagal baca DB: {e}")
        return

    if df.empty:
        print("   ✅ Tidak ada berita baru untuk diproses.")
        return

    print(f"   📦 Memproses {len(df)} berita baru...")

    # 2. Load Model AI (IndoBERT)
    print("   🤖 Sedang memuat model AI (IndoBERT)...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="w11wo/indonesian-roberta-base-sentiment-classifier",
        tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier"
    )

    # 3. Proses Loop
    processed_data = []

    for _, row in df.iterrows():
        text_to_analyze = row['title']  # Kita analisis Judulnya
        text_lower = text_to_analyze.lower()

        # --- 🚨 FILTER RELEVANSI (SATPAM KAMPUS) ---
        # Jika judul TIDAK mengandung kata unsika ATAU singaperbangsa, langsung hapus!
        if not ('unsika' in text_lower or 'singaperbangsa' in text_lower):
            try:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM news_raw WHERE id = :id"), {"id": row['id']})
                print(f"   🗑️ HAPUS (Berita Nyasar): '{text_to_analyze[:45]}...'")
            except Exception as e:
                print(f"   ⚠️ Gagal menghapus berita nyasar: {e}")

            # Lewati prediksi sentimen, langsung lanjut ke baris berita berikutnya
            continue
        # --------------------------------------------

        # Prediksi AI Murni
        result = sentiment_pipeline(text_to_analyze)[0]
        label = result['label']
        score = result['score']

        # --- TERAPKAN LOGIKA CERDAS DI SINI ---
        final_label, final_score = refine_sentiment(text_to_analyze, label, score)

        # Mapping label model ke standar kita
        final_label = final_label.capitalize()

        processed_data.append({
            'news_id': row['id'],
            'clean_text': clean_text(text_to_analyze),
            'sentiment_label': final_label,
            'sentiment_score': final_score,
            'processed_at': datetime.now()
        })

        # Print preview biar kelihatan bedanya
        if label.capitalize() != final_label:
            print(f"   ✨ FIX: '{text_to_analyze[:30]}...' | AI: {label} -> SMART: {final_label}")
        else:
            print(f"   Process: '{text_to_analyze[:30]}...' -> {final_label}")

    # 4. Simpan ke Database
    if processed_data:
        df_result = pd.DataFrame(processed_data)
        df_result.to_sql('news_processed', engine, if_exists='append', index=False)
        print(f"   ✅ Sukses menyimpan {len(df_result)} hasil analisis.")

    print("🏁 Job Selesai.")


if __name__ == "__main__":
    run_news_processing()