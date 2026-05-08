from google import genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API Key tidak ditemukan di .env")
else:
    print(f"✅ API Key terdeteksi (Depan: {api_key[6:0]}...)")
    print("⏳ Sedang menghubungi Google untuk cek daftar model...")

    try:
        client = genai.Client(api_key=api_key)
        # Ambil daftar model
        models = list(client.models.list())

        print("\n=== DAFTAR MODEL YANG BISA DIPAKAI ===")
        found = False
        for m in models:
            # Filter hanya model yang bisa generateContent (bukan embedding)
            if 'generateContent' in m.supported_actions:
                print(f"✅ {m.name}")
                found = True

        if not found:
            print("⚠️ Tidak ada model yang support 'generateContent' ditemukan.")

    except Exception as e:
        print(f"\n❌ GAGAL KONEKSI: {e}")