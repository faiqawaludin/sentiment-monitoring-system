import os
from dotenv import load_dotenv

from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di file .env!")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-2.5-flash"

def generate_summary(text_data, context_type="Berita"):
    if not text_data or len(text_data) == 0:
        return f"Belum ada data {context_type} yang cukup untuk dianalisis hari ini."

    data_terbatas = text_data[:15]  # Naikkan sedikit agar konteks lebih kaya
    gabungan_teks = " | ".join([str(t)[:200] for t in data_terbatas if str(t).strip() != ""])

    prompt = f"""Kamu adalah seorang PR Analyst senior di Universitas Singaperbangsa Karawang (UNSIKA) dengan pengalaman lebih dari 10 tahun dalam manajemen reputasi institusi pendidikan tinggi.

Tugasmu adalah menganalisis kumpulan data {context_type} terbaru berikut dan menyusun laporan analisis yang komprehensif.

DATA {context_type.upper()} TERBARU:
{gabungan_teks}

Susun laporanmu dengan struktur berikut:

**📊 Ringkasan Eksekutif**
Tuliskan gambaran umum situasi dalam 2-3 kalimat. Apa yang sedang terjadi secara keseluruhan?

**🎯 Sentimen Dominan**
- Tentukan apakah sentimen keseluruhan: Positif / Negatif / Netral / Campuran
- Jelaskan alasan dan proporsinya secara singkat
- Sebutkan faktor utama yang mendorong sentimen tersebut

**🔍 Temuan Kunci**
Sebutkan 3-5 poin temuan paling penting dari data ini dalam format bullet point. Setiap poin harus actionable dan spesifik.

**⚠️ Potensi Risiko Reputasi**
Identifikasi 2-3 hal yang berpotensi menjadi ancaman reputasi UNSIKA jika tidak ditangani. Jika tidak ada risiko signifikan, nyatakan demikian.

**💡 Rekomendasi Strategis**
Berikan 2-3 langkah konkret yang sebaiknya diambil tim PR UNSIKA berdasarkan temuan di atas.

ATURAN KETAT:
- Gunakan bahasa Indonesia yang profesional namun mudah dipahami
- Hanya analisis berdasarkan data yang diberikan, JANGAN mengarang informasi
- Jika data terlalu sedikit untuk suatu seksi, tulis "Data belum cukup untuk analisis bagian ini"
- Gunakan format markdown agar mudah dibaca"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gagal menghubungi AI Gemini: {str(e)}"

def generate_hot_topics(news_titles, tweet_texts):
    news_terbatas = news_titles[:12]
    tweet_terbatas = tweet_texts[:12]

    teks_gabungan = (
        "BERITA:\n" + "\n".join([f"- {t}" for t in news_terbatas]) +
        "\n\nCUITAN TWITTER/X:\n" + "\n".join([f"- {t}" for t in tweet_terbatas])
    )

    prompt = f"""Kamu adalah PR Analyst senior UNSIKA (Universitas Singaperbangsa Karawang) yang bertugas memantau isu publik secara real-time.

Berikut adalah data terbaru dari media berita dan Twitter/X hari ini:

{teks_gabungan}

Analisis data di atas dan susun laporan Hot Topics dengan format berikut:

**🔥 5 Topik Terpanas Hari Ini**

Untuk setiap topik, gunakan format:

**[Nomor]. [Nama Topik]**
- 📌 **Isu:** [Deskripsi singkat isu dalam 1-2 kalimat]
- 📣 **Sumber Dominan:** [Berita / Twitter / Keduanya]
- 💬 **Sentimen Publik:** [Positif / Negatif / Netral] — [alasan singkat]
- 🎯 **Urgensi untuk UNSIKA:** [Tinggi / Sedang / Rendah] — [alasan singkat]

---

Setelah daftar 5 topik, tambahkan:

**📝 Kesimpulan Harian**
Tuliskan 1 paragraf singkat tentang gambaran keseluruhan hari ini: apakah ada tren, pola, atau hal yang perlu diwaspadai oleh tim PR UNSIKA.

ATURAN KETAT:
- Hanya gunakan informasi dari data yang diberikan
- Jika data tidak cukup untuk 5 topik, sebutkan hanya yang relevan
- Gunakan bahasa Indonesia profesional
- Format harus rapi dengan markdown"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gagal memuat topik AI: {str(e)}"