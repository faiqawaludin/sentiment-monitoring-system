REMOSY (Reputation Monitoring System)

**REMOSY** adalah sistem cerdas pemantauan reputasi institusi berbasis kecerdasan buatan (AI) yang dikembangkan khusus untuk memonitor opini publik, berita media massa, dan cuitan media sosial terkait Institusi. Proyek ini mengimplementasikan *end-to-end automated data pipeline*, mulai dari ekstraksi data mentah hingga visualisasi analitik interaktif, untuk membantu pemangku kepentingan dalam pengambilan keputusan strategis (*data-driven decision making*).

## ✨ Fitur Utama

- **Automated Data Scraping:** Pipeline otomatis untuk mengekstraksi data dari portal berita (Google News) dan media sosial (X/Twitter).
- **Smart Sentiment Analysis (Reputation-Aware):** Menggunakan model AI **IndoBERT** yang dikombinasikan dengan *Hybrid Logic* dan "Filter Satpam Kampus" untuk mendeteksi sentimen positif/negatif yang spesifik terhadap konteks reputasi institusi.
- **AI-Powered Topic Modeling:** Menggabungkan algoritma *Latent Dirichlet Allocation (LDA)* dengan **Google Gemini AI** untuk mengekstrak, merangkum, dan menganalisis 5 isu terhangat secara otomatis.
- **Interactive Executive Dashboard:** Antarmuka visual yang dibangun dengan **Streamlit** dan **Plotly**, dilengkapi dengan fitur *Cross-Filtering* untuk analisis data yang mendalam.
- **Tanya AI (RAG Chatbot):** Asisten virtual cerdas yang terintegrasi langsung ke dalam *dashboard*, memungkinkan pengguna untuk "mengobrol" dengan data kampus secara *real-time*.

---

Teknologi & Arsitektur

Proyek ini dibangun menggunakan kombinasi teknologi *Data Engineering*, *Machine Learning*, dan *Web Development*:
* **Bahasa Pemrograman:** Python
* **Database:** PostgreSQL (Relational Database dengan struktur terpisah untuk `raw` dan `processed` data)
* **NLP & AI:** HuggingFace Transformers (IndoBERT), Scikit-Learn (LDA), Sastrawi, Google Gemini API
* **Data Processing & ETL:** Pandas, SQLAlchemy, Regex
* **Dashboard & Visualisasi:** Streamlit, Plotly Express, WordCloud

Metodologi
Pengembangan sistem ini mengadopsi integrasi kerangka kerja **CRISP-DM** (Cross-Industry Standard Process for Data Mining) dan **Scrum** untuk mengakomodasi siklus pengembangan *Machine Learning* yang iteratif dan dinamis.
