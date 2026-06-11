# REMOSY (Reputation Monitoring System)

**REMOSY** adalah sistem cerdas pemantauan reputasi institusi berbasis kecerdasan buatan (AI) yang dirancang dengan filosofi **Zero Human Intervention**. Sistem ini mengintegrasikan arsitektur *Modern Data Stack* (MDS) untuk mengotomatisasi seluruh siklus pemantauan media massa dan media sosial secara proaktif, terstruktur, dan *real-time*. 

Proyek ini berhasil mengimplementasikan *end-to-end automated data pipeline* paralel tingkat tinggi, mulai dari tahap rekayasa data (*data engineering*), pemodelan *Deep Learning*, hingga penyajian wawasan eksekutif (*executive insights*) berbasis *Generative AI* untuk mendukung pengambilan keputusan strategis (*data-driven decision making*) oleh unit *Public Relations* atau Humas institusi.

---

## Fitur Utama & Inovasi Sistem

- **Multi-Source Automated Data Ingestion:** *Pipeline* penarikan data terotomatisasi secara harian dari dua kanal utama, yaitu Google News RSS dan platform X (Twitter). Dilengkapi dengan parameter *Smart Scraper* (`when:1y`) untuk mencegah *data decay* (berita usang) serta mekanisme penapisan *check-sum* URL untuk eliminasi data duplikat secara *real-time*.
- **Orkestrasi Pipeline End-to-End (Apache Airflow):** Pengondisian seluruh alur kerja ETL (Extract, Transform, Load) dikendalikan secara terpusat oleh **Apache Airflow** melalui topologi *Directed Acyclic Graphs* (DAGs) paralel yang berjalan otomatis tanpa pemicu manual.
- **Smart Sentiment Analysis (Hybrid IndoBERT):** Lapisan analisis sentimen menggunakan arsitektur *Deep Learning* berbasis **IndoBERT** yang dikombinasikan dengan metode *Hybrid Logic* (Kamus Kata Kunci) dan aturan *Reputation-Aware* (Filter Isu Krisis Kampus) untuk menjamin akurasi pelabelan teks informal khas Indonesia.
- **Generative AI Executive Synthesis (Pengganti LDA):** Meninggalkan algoritma klasterisasi tradisional, sistem ini mengintegrasikan **Google Gemini 2.5-Flash API** pada level *backend orchestration* untuk melakukan pre-komputasi naratif harian yang menghasilkan fitur **" 5 Isu Terhangat"** dan **"Ringkasan Eksekutif (TL;DR)"** secara instan.
- **Asisten AI Interaktif (RAG Chatbot System):** Fitur asisten cerdas berbasis *Retrieval-Augmented Generation* (RAG) yang ditenagai oleh **FAISS Vector Database** dan *HuggingFace Multilingual Embeddings*. Memungkinkan tim Humas untuk melakukan tanya-jawab langsung dengan basis data dokumen institusi secara kontekstual tanpa risiko halusinasi AI.
- **Enterprise Executive Dashboard:** Antarmuka visual dinamis berbasis **Streamlit** dan **Plotly** (Custom Dark Mode) yang dilengkapi fitur **Cross-Filtering** (grafik berfungsi sebagai filter tabel data mentah secara dinamis) serta tautan interaktif langsung ke sumber data asli (`LinkColumn`).

---

##  Arsitektur Data & Medallion Design

Sistem ini diisolasi penuh di dalam lingkungan **Docker (Containerization)** dan mengadopsi prinsip **Medallion Architecture** di atas database internal untuk menjamin tata kelola kualitas data:

1.  **Bronze Layer (Raw Data):** Penyimpanan data mentah langsung dari *scraper* pada tabel `news_raw` dan `tweets_raw`.
2.  **Silver Layer (Enriched Data):** Tahap di mana Airflow mengeksekusi *text preprocessing* dan inferensi model IndoBERT, menghasilkan tabel bersih `news_processed` dan `tweets_processed` yang telah diperkaya kolom label serta skor sentimen.
3.  **Gold Layer (Serving Data):** Data agregasi yang siap dikonsumsi instan oleh Streamlit, data indeks ruang vektor pada **FAISS Store** (`index.faiss`), serta tabel *cache* analitik AI (`ai_analysis_cache`) untuk mengeliminasi latensi pemanggilan LLM (*Zero-Latency Front-End Execution*).

---

##  Tech Stack yang Digunakan

* **Data Engineering & Orchestration:** Apache Airflow, Docker, Docker Compose
* **Storage & Data Warehouse:** PostgreSQL (Containerized Lokal, dipilih untuk efisiensi biaya infrastruktur akademik dan eliminasi latensi jaringan cloud)
* **Natural Language Processing & AI:** HuggingFace Transformers (IndoBERT), Google GenAI SDK (Gemini 2.5-Flash), LangChain Core & Community, FAISS (Facebook AI Similarity Search), Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Data Processing & Analytics:** Pandas, SQLAlchemy, Regex, PyArrow
* **Visualization & Frontend UI:** Streamlit, Streamlit Option Menu, Plotly Express, WordCloud

---

##  Metodologi Pengembangan

Pengembangan sistem REMOSY mengadopsi integrasi harmonis antara:
* **CRISP-DM (Cross-Industry Standard Process for Data Mining):** Mengawal siklus analitik data mulai dari *Business Understanding*, *Data Understanding*, *Data Preparation*, *Modeling*, *Evaluation*, hingga *Deployment*.
* **Agile Scrum Framework:** Mengakomodasi pengembangan perangkat lunak
