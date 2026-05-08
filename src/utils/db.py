from sqlalchemy import create_engine
import os


def get_db_engine():
    """
    Membuat koneksi ke PostgreSQL Data Warehouse.
    Menggunakan localhost jika dijalankan di laptop,
    atau nama service docker jika dijalankan di dalam container.
    """
    # Default config untuk Laptop (Localhost)
    DB_USER = os.getenv("DB_USER", "remosy_user")
    DB_PASS = os.getenv("DB_PASS", "remosy_password")
    DB_NAME = os.getenv("DB_NAME", "remosy_dw")

    # Deteksi apakah kita di dalam Docker atau di Laptop
    # Kalau di laptop pakai localhost:5434, kalau di docker pakai postgres-dw:5432
    if os.path.exists('/.dockerenv'):
        DB_HOST = "postgres-dw"
        DB_PORT = "5432"
    else:
        DB_HOST = "localhost"
        DB_PORT = "5434"

    url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)