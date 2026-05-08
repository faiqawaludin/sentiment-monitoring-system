from transformers import pipeline

MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"

print(" Sedang memuat model AI (IndoBERT)...")
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)
print(" Model AI siap digunakan!")

# --- KAMUS KATA KUNCI (Hybrid Method) ---
# Gunakan ini untuk "memancing" sentimen berita yang terlalu formal
POSITIVE_KEYWORDS = [
    "juara", "menang", "prestasi", "penghargaan", "terbaik",
    "sukses", "berhasil", "maju", "unggul", "apresiasi",
    "bangga", "keren", "inovasi", "kontribusi", "positif",
    "mendukung", "membantu", "beasiswa", "lulus", "wisuda", "teknologi", "mitigasi", "tepat", "tepat guna"
]

NEGATIVE_KEYWORDS = [
    "korupsi", "mangkrak", "demo", "protes", "gagal",
    "kecewa", "batal", "rugi", "sanksi", "pelanggaran",
    "tersangka", "pidana", "polisi", "kecelakaan", "banjir",
    "sampah", "bau", "kotor", "macet", "mahal", "pungli"
]


def predict_sentiment(text):
    """
    Prediksi sentimen dengan metode Hybrid: AI + Keyword Matching.
    """
    if not text or not isinstance(text, str):
        return "Neutral", 0.0

    try:
        text = text[:512]
        result = sentiment_pipeline(text)[0]

        label = result['label']
        score = result['score']

        # Standardisasi Label AI
        if 'positive' in label.lower():
            final_label = 'Positive'
        elif 'negative' in label.lower():
            final_label = 'Negative'
        else:
            final_label = 'Neutral'

        # --- LOGIKA HYBRID (OVERRIDE) ---
        # Jika AI bilang Neutral, kita cek apakah ada kata kunci penting yang terlewat
        if final_label == 'Neutral':
            text_lower = text.lower()

            # Cek Negatif dulu (Prioritas peringatan)
            if any(word in text_lower for word in NEGATIVE_KEYWORDS):
                final_label = 'Negative'
                score = 0.85  # Kita beri skor manual tinggi

            # Cek Positif
            elif any(word in text_lower for word in POSITIVE_KEYWORDS):
                final_label = 'Positive'
                score = 0.85

        return final_label, score

    except Exception as e:
        print(f"️ Gagal prediksi: {e}")
        return "Neutral", 0.0


# --- BLOK TESTING ---
if __name__ == "__main__":
    print("\n MEMULAI UJI COBA HYBRID...\n")

    # Contoh Berita yang tadinya Neutral
    contoh = [
        "mahasiswa unsika raih juara 1 lomba internasional",  # Ada kata 'juara'
        "rektor meresmikan gedung baru",  # Faktual (tetap neutral)
        "mahasiswa protes ukt mahal di depan rektorat"  # Ada kata 'protes' & 'mahal'
    ]

    for t in contoh:
        l, s = predict_sentiment(t)
        print(f" Teks: {t}")
        print(f" Hasil: {l} ({s:.2f})\n")