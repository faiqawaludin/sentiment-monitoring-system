import sys
import os
import pandas as pd
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '../..'))
if root_project not in sys.path:
    sys.path.append(root_project)

load_dotenv()

from src.utils.db import get_db_engine
from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "remosy_vectors"


def build_pgvector_index():
    print("Connecting to Database")
    engine = get_db_engine()

    # Rakit connection string yang ANTI-NYASAR
    pg_db = os.getenv("DB_NAME", "remosy_dw")
    pg_user = os.getenv("DB_USER", "remosy_user")
    pg_pass = os.getenv("DB_PASSWORD", "remosy_password")

    # Samakan dengan logika cerdas di db.py
    if os.path.exists('/.dockerenv'):
        pg_host = "postgres-dw"
        pg_port = "5432"
    else:
        pg_host = "localhost"
        pg_port = "5434"

    db_url = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

    print("Ambil Data")

    # ==========================================
    # A. Tarik Data Berita
    # ==========================================
    df_news = pd.DataFrame()
    try:
        query_news = """
            SELECT nr.title          AS teks,
                   nr.source,
                   nr.published_date AS tanggal,
                   np.sentiment_label,
                   np.sentiment_score,
                   'Berita'          AS platform
            FROM news_raw nr
                     JOIN news_processed np ON nr.id = np.news_id
            WHERE np.sentiment_label IS NOT NULL
            ORDER BY nr.scraped_at DESC
        """
        df_news = pd.read_sql(query_news, engine)
        print(f"Berhasil menarik {len(df_news)} data Berita.")
    except Exception as e:
        print(f"Tabel Berita error: {e}")

    # ==========================================
    # B. Tarik Data Twitter
    # ==========================================
    df_twitter = pd.DataFrame()
    try:
        query_twitter = """
            SELECT tr.full_text  AS teks,
                   tr.username   AS source,
                   tr.scraped_at AS tanggal,
                   tp.sentiment_label,
                   tp.sentiment_score,
                   'Twitter'     AS platform
            FROM tweets_raw tr
                     JOIN tweets_processed tp ON tr.id = tp.tweet_id
            WHERE tp.sentiment_label IS NOT NULL
              AND tr.scraped_at >= NOW() - INTERVAL '365 days'
            ORDER BY tr.scraped_at DESC
        """
        df_twitter = pd.read_sql(query_twitter, engine)
        print(f"Berhasil menarik {len(df_twitter)} data Twitter.")
    except Exception as e:
        print(f"Gagal menarik data Twitter. Error: {e}")

    # ==========================================
    # C. Gabungkan Semua Data
    # ==========================================
    df_data = pd.concat([df_news, df_twitter], ignore_index=True)

    if df_data.empty:
        print("Semua data kosong! Vector DB batal dibuat.")
        return

    # ==========================================
    # D. Konversi ke Dokumen LangChain
    # ==========================================
    print(f"Memproses {len(df_data)} data gabungan menjadi narasi AI...")
    documents = []
    for _, row in df_data.iterrows():
        date_str = pd.to_datetime(row['tanggal']).strftime('%d %B %Y') if pd.notnull(
            row['tanggal']) else 'Tanggal tidak diketahui'

        narasi = (f"Pada tanggal {date_str}, terdapat pantauan {row['platform']} dari {row['source']} "
                  f"berisi: '{row['teks']}'. Sentimen: {row['sentiment_label']} (Skor: {row['sentiment_score']}).")

        metadata = {
            "source":    str(row['source']),
            "sentiment": str(row['sentiment_label']),
            "platform":  str(row['platform']),
            "date":      date_str
        }

        documents.append(Document(page_content=narasi, metadata=metadata))

    # ==========================================
    # E. Load Embedding Model
    # ==========================================
    print("Memuat Model AI Multilingual (Lokal & Gratis)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ==========================================
    # F. Simpan ke pgvector via LangChain
    # ==========================================
    print("🧬 Menyimpan vektor ke PostgreSQL (pgvector) via langchain_pg_embedding...")

    PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=db_url,
        pre_delete_collection=True,
    )

    print(f"SUKSES! Vector Database berhasil disimpan ke tabel "
          f"langchain_pg_embedding dengan collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_pgvector_index()