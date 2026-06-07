import sys
import os
import pandas as pd
from dotenv import load_dotenv

# Konfigurasi Path agar bisa import utils dari root project
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '../..'))
if root_project not in sys.path:
    sys.path.append(root_project)

from src.utils.db import get_db_engine
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def build_faiss_index():
    print("⏳ Menyiapkan koneksi ke Database...")
    engine = get_db_engine()
    load_dotenv()

    print("📥 Menyedot data sentimen dari PostgreSQL...")

    # ==========================================
    # A. Tarik Data Berita
    # ==========================================
    try:
        query_news = """
                     SELECT nr.title          as teks, \
                            nr.source, \
                            nr.published_date as tanggal, \
                            np.sentiment_label, \
                            np.sentiment_score, \
                            'Berita'          as platform
                     FROM news_raw nr
                              JOIN news_processed np ON nr.id = np.news_id
                     WHERE np.sentiment_label IS NOT NULL \
                     """
        df_news = pd.read_sql(query_news, engine)
        print(f"✅ Berhasil menarik {len(df_news)} data Berita.")
    except Exception as e:
        print(f"⚠️ Tabel Berita error: {e}")
        df_news = pd.DataFrame()  # Jaring pengaman

    # ==========================================
    # B. Tarik Data Twitter
    # ==========================================
    try:
        query_twitter = """
                        SELECT tr.full_text  as teks,
                               tr.username   as source,
                               tr.scraped_at as tanggal,
                               tp.sentiment_label,
                               tp.sentiment_score,
                               'Twitter'     as platform
                        FROM tweets_raw tr
                                 JOIN tweets_processed tp ON tr.id = tp.tweet_id
                        WHERE tp.sentiment_label IS NOT NULL \
                        """
        df_twitter = pd.read_sql(query_twitter, engine)
        print(f"✅ Berhasil menarik {len(df_twitter)} data Twitter.")
    except Exception as e:
        print(f"⚠️ Gagal menarik data Twitter. Error: {e}")
        df_twitter = pd.DataFrame()  # Jaring pengaman mutlak agar tidak UnboundLocalError

    # ==========================================
    # C. Gabungkan Semua Data
    # ==========================================
    df_data = pd.concat([df_news, df_twitter], ignore_index=True)

    if df_data.empty:
        print("❌ Semua data kosong! Vector DB batal dibuat.")
        return

    # 2. UBAH DATA TABEL MENJADI DOKUMEN TEKS UNTUK AI
    print(f"🧠 Memproses {len(df_data)} data gabungan menjadi narasi AI...")
    documents = []
    for _, row in df_data.iterrows():
        date_str = pd.to_datetime(row['tanggal']).strftime('%d %B %Y') if pd.notnull(
            row['tanggal']) else 'Tanggal tidak diketahui'

        narasi = (f"Pada tanggal {date_str}, terdapat pantauan {row['platform']} dari {row['source']} "
                  f"berisi: '{row['teks']}'. Sentimen: {row['sentiment_label']} (Skor: {row['sentiment_score']}).")

        metadata = {
            "source": row['source'],
            "sentiment": row['sentiment_label'],
            "platform": row['platform'],
            "date": date_str
        }

        doc = Document(page_content=narasi, metadata=metadata)
        documents.append(doc)

    # 3. UBAH TEKS MENJADI VEKTOR (EMBEDDINGS) MENGGUNAKAN HUGGINGFACE
    print("🚀 Memuat Model AI Multilingual (Lokal & Gratis)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("🧬 Sedang melakukan konversi narasi ke Vektor (Mungkin butuh waktu beberapa detik)...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Simpan ke folder lokal
    save_path = os.path.join(root_project, "data", "faiss_index")
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

    print(f"✅ SUKSES BESAR! Vector Database berhasil disimpan di: {save_path}")


if __name__ == "__main__":
    build_faiss_index()