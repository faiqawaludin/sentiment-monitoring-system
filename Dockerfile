# Gunakan image dasar Airflow
FROM apache/airflow:2.10.0

# Ganti ke root untuk install driver sistem, Node.js, dan Library Browser
USER root
RUN apt-get update && \
    apt-get install -y \
    libpq-dev \
    gcc \
    nodejs \
    npm \
    # --- Dependencies untuk Chromium/Playwright ---
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libgtk-3-0 \
    # ---------------------------------------------
    && apt-get clean

# Kembali ke user airflow
USER airflow

# Instalasi Library Python
# 1. Install PyTorch versi CPU (Ringan)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install library sisanya (Perhatikan baris 'RUN' di bawah ini jangan sampai hilang)
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    beautifulsoup4 \
    lxml \
    requests \
    sqlalchemy \
    psycopg2-binary \
    plotly \
    streamlit \
    xlsxwriter \
    transformers \
    huggingface_hub \
    wordcloud \
    matplotlib \
    streamlit-option-menu \
    scikit-learn \
    sastrawi \
    python-dotenv \
    google-genai