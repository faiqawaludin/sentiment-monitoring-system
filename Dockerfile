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

# ==========================================
# 2. CARA CERDAS: BACA DARI REQUIREMENTS.TXT
# ==========================================
# Salin requirements dari laptop ke dalam kontainer
COPY requirements-local.txt /requirements.txt


# Install SEMUA library sekaligus dari file teks tersebut
RUN pip install --no-cache-dir -r /requirements.txt