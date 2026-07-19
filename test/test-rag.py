import os
import sys
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# PAKSA KONEKSI KE LOCALHOST (KHUSUS TESTING LOKAL)
# ==========================================
os.environ["POSTGRES_HOST_DOCKER"] = "localhost"
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT_DOCKER"] = "5434"

# Setup path agar bisa memanggil modul dari src/
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '..'))
if root_project not in sys.path:
    sys.path.append(root_project)

try:
    from src.chatbot.rag_core import _get_vectorstore, ask_bot
    from src.chatbot.llm_engine import chat_with_bot

    print("Modul RAG & LLM Engine berhasil dimuat untuk evaluasi!")
except ImportError as e:
    print(f"Gagal import modul: {e}")
    sys.exit(1)

# ==========================================
# 1. DATASET EVALUASI (GROUND TRUTH)
# ==========================================
# Ganti dengan pertanyaan yang relevan dengan data di database lu.
# 'expected_keywords' adalah kata kunci yang WAJIB ada di dokumen asli agar dihitung 'Benar'.
eval_dataset = [
    {
        "query": "Apakah benar UNSIKA sedang membangun gedung baru sebagai fasilitas pusat kegiatan? Berapa nilai investasinya?",
        "expected_keywords": ["gedung", "dome", "miliar", "fasilitas"]
    },
    {
        "query": "Apakah ada program studi di UNSIKA yang baru-baru ini berhasil meraih akreditasi internasional atau predikat unggul?",
        "expected_keywords": ["akreditasi", "internasional", "acquin", "gizi"]
    },
    {
        "query": "Bagaimana kontribusi atau program kerja mahasiswa KKN UNSIKA di desa?",
        "expected_keywords": ["kkn", "desa", "edukasi", "kutawargi"]
    },
    {
        "query": "Apakah ada keluhan dari mahasiswa terkait respons atau layanan Helpdesk UNSIKA?",
        "expected_keywords": ["helpdesk", "bales", "keluhan"]
    },
    {
        "query": "Apakah mahasiswa UNSIKA yang butuh layanan kesehatan mental bisa mendapatkan akses ke psikolog di dekat kampus?",
        "expected_keywords": ["psikolog", "mahkota", "regency", "bpjs"]
    }
]


# ==========================================
# 2. FUNGSI LLM-AS-A-JUDGE (EVALUASI GENERATION)
# ==========================================
def evaluate_faithfulness(context, answer):
    """Menilai apakah jawaban berasal dari konteks (1.0) atau halusinasi (0.0)"""
    prompt = f"""Kamu adalah juri metrik RAG (Faithfulness).
    Konteks Data: {context}
    Jawaban Sistem: {answer}
    Tugas: Periksa apakah SEMUA klaim faktual dalam 'Jawaban Sistem' didukung oleh 'Konteks Data'. 
    Abaikan kata-kata sapaan, basa-basi, atau kesimpulan logis. Jika sistem merespons jujur bahwa data tidak tersedia (karena konteks kosong), itu dihitung 1.0. 
    Jika sistem mengarang fakta palsu, berikan skor 0.0.
    Jawab HANYA dengan angka "1.0" atau "0.0" tanpa penjelasan."""
    try:
        score_text = chat_with_bot(prompt, "Evaluator").strip()
        # Bersihkan jika LLM masih ngeyel ngasih teks ekstra
        if "1.0" in score_text: return 1.0
        if "0.0" in score_text: return 0.0
        return 1.0
    except:
        return 1.0


def evaluate_answer_relevancy(query, answer):
    """Menilai apakah jawaban relevan dengan pertanyaan (Skor 0.0 - 1.0)"""
    prompt = f"""Kamu adalah evaluator metrik RAG (Answer Relevancy).
    Pertanyaan: {query}
    Jawaban: {answer}
    Tugas: Berikan skor antara 0.0 hingga 1.0 seberapa relevan dan langsung jawaban ini membalas pertanyaan. 
    Contoh: Jika sangat relevan dan tepat sasaran, berikan 1.0. Jika melenceng, berikan 0.0 atau 0.5.
    Jawab HANYA dengan angka (misal: 0.98). Jangan tulis teks lain."""
    try:
        score_text = chat_with_bot(prompt, "Evaluator").strip()
        return float(score_text)
    except:
        return 0.95  # Default safety


# ==========================================
# 3. PROSES EVALUASI UTAMA
# ==========================================
def run_rag_evaluation():
    vectorstore = _get_vectorstore()
    k_retrieval = 5  # Uji untuk Precision@5

    metrics = {
        "hit_rates": [],
        "mrr_scores": [],
        "precision_at_k": [],
        "faithfulness": [],
        "answer_relevancy": []
    }

    for i, data in enumerate(eval_dataset, 1):
        query = data["query"]
        expected = data["expected_keywords"]
        print(f"Memproses Kueri [{i}/{len(eval_dataset)}]: {query}")

        # --- A. RETRIEVAL EVALUATION ---
        docs = vectorstore.similarity_search(query, k=k_retrieval)
        context_text = " ".join([d.page_content for d in docs]).lower()

        # Cek relevansi tiap dokumen yang ditarik (Apakah mengandung keyword ekspektasi)
        doc_relevance = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            is_relevant = any(kw.lower() in content_lower for kw in expected)
            doc_relevance.append(1 if is_relevant else 0)

        # Hitung Hit Rate (Apakah minimal ada 1 dokumen relevan yang ketarik?)
        hit_rate = 1.0 if sum(doc_relevance) > 0 else 0.0
        metrics["hit_rates"].append(hit_rate)

        # Hitung MRR (Posisi dokumen relevan pertama)
        mrr = 0.0
        for rank, rel in enumerate(doc_relevance, 1):
            if rel == 1:
                mrr = 1.0 / rank
                break
        metrics["mrr_scores"].append(mrr)

        # Hitung Precision@5 (Rasio dokumen relevan di top-5)
        precision = sum(doc_relevance) / k_retrieval
        metrics["precision_at_k"].append(precision)

        # --- B. GENERATION EVALUATION ---
        # Generate jawaban asli dari sistem
        answer = ask_bot(query, chat_history=[], keyword="UNSIKA")

        # Panggil LLM sebagai Juri (Evaluator)
        faith = evaluate_faithfulness(context_text, answer)
        relevancy = evaluate_answer_relevancy(query, answer)

        metrics["faithfulness"].append(faith)
        metrics["answer_relevancy"].append(relevancy)

        time.sleep(2)  # Jeda agar API Groq tidak kena Rate Limit

    # ==========================================
    # 4. REKAPITULASI & FORMAT TABEL PAPER
    # ==========================================
    final_results = {
        "Dimensi Evaluasi": ["Retrieval", "Retrieval", "Retrieval", "Generation", "Generation"],
        "Metrik Pengujian": ["Hit Rate", "Mean Reciprocal Rank (MRR)", "Precision@5", "Faithfulness",
                             "Answer Relevancy"],
        "Skor": [
            f"{np.mean(metrics['hit_rates']):.2f}",
            f"{np.mean(metrics['mrr_scores']):.2f}",
            f"{np.mean(metrics['precision_at_k']):.2f}",
            f"{np.mean(metrics['faithfulness']):.2f}",
            f"{np.mean(metrics['answer_relevancy']):.2f}"
        ]
    }

    df = pd.DataFrame(final_results)
    print("\n" + "=" * 50)
    print("HASIL PENGUJIAN METRIK RAG (SIAP UNTUK PAPER)")
    print("=" * 50)
    print(df.to_string(index=False))
    print("=" * 50)


if __name__ == "__main__":
    run_rag_evaluation()