import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di file .env!")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')


def generate_summary(text_data, context_type="Berita"):
    """Fungsi untuk merangkum teks menjadi Executive Summary."""
    if not text_data or len(text_data) == 0:
        return f"Belum ada data {context_type} yang cukup untuk dianalisis hari ini."

    gabungan_teks = " | ".join([str(t) for t in text_data if str(t).strip() != ""])

    prompt = f"""
    Kamu adalah seorang Public Relations Analyst yang bekerja untuk Universitas Singaperbangsa Karawang (UNSIKA).
    Tugasmu adalah membuat ringkasan eksekutif (TL;DR) berdasarkan kumpulan data {context_type} terbaru berikut ini.

    DATA {context_type.upper()}:
    {gabungan_teks}

    ATURAN:
    1. Buat ringkasan HANYA dalam 1 paragraf (Maksimal 3 kalimat).
    2. Gunakan bahasa Indonesia yang profesional namun mudah dipahami.
    3. Soroti sentimen utama (apakah sedang banyak pujian, atau ada keluhan/kasus).
    4. Jika data berisi info pendaftaran mahasiswa baru, sebutkan secara singkat.
    5. JANGAN mengarang informasi di luar teks yang diberikan.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gagal menghubungi AI Gemini: {str(e)}"