import re
import html

def clean_text(text):
    if not text:
        return ""

    # 1. Decode HTML entities (misal: &amp; -> &)
    text = html.unescape(text)

    # 2. Lowercase (Huruf kecil semua agar seragam)
    text = text.lower()

    # 3. Hapus URL / Link (http://... atau https://...)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 4. Hapus Mention (@username) - Khusus untuk Twitter nanti
    text = re.sub(r'@\w+', '', text)

    # 5. Hapus Hashtag (#) tapi biarkan kata-katanya (Opsional, di sini kita hapus simbolnya saja)
    text = re.sub(r'#', '', text)

    # 6. Hapus Karakter selain Huruf, Angka, dan Spasi (Emoji & Simbol aneh hilang)
    text = re.sub(r'[^\w\s]', ' ', text)

    # 7. Hapus Angka (Opsional: sentimen biasanya tidak butuh angka)
    text = re.sub(r'\d+', '', text)

    # 8. Hapus Spasi Berlebih (Double space akibat penghapusan di atas)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- BLOK TESTING (Hanya jalan kalau file ini di-Run langsung) ---
if __name__ == "__main__":
    print(" MEMULAI PEMBERSIHAN TEKS...\n")

    # Contoh 1: Berita Kotor
    berita_kotor = "BREAKING NEWS: Rektor UNSIKA meresmikan Gedung Baru!! (Baca selengkapnya: https://bit.ly/news)"
    print(f" Asli News   : {berita_kotor}")
    print(f" Bersih News : {clean_text(berita_kotor)}\n")

    # Contoh 2: Tweet Kotor
    tweet_kotor = "RT @MahasiswaUnsika: Wah parah sih semester ini.. 😭 #UnsikaJaya #2026 100% capek!"
    print(f" Asli Tweet  : {tweet_kotor}")
    print(f" Bersih Tweet: {clean_text(tweet_kotor)}")