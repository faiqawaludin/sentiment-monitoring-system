import os
import sys
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '../..'))
if root_project not in sys.path:
    sys.path.append(root_project)

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"

@st.cache_resource(show_spinner=False)
def get_vector_db():
    """Memuat Vector Database ke RAM dan menguncinya (Cache)"""
    print("🚀 Memuat Vector DB ke Memori...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    save_path = os.path.join(root_project, "data", "faiss_index")
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


def ask_bot(query: str, chat_history: list = None):
    """
    Menjawab pertanyaan menggunakan RAG (Retrieval-Augmented Generation).

    Args:
        query: Pertanyaan dari pengguna
        chat_history: List of dict [{"role": "user/assistant", "content": "..."}]
                      untuk konteks percakapan sebelumnya (opsional)
    """
    vectorstore = get_vector_db()

    # Retrieve dokumen relevan (Hemat token dengan k=5)
    docs = vectorstore.similarity_search(query, k=5)

    # Format konteks dengan metadata jika tersedia
    context_parts = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source_info = ""
        if metadata.get("source"):
            source_info += f"[Sumber: {metadata['source']}]"
        if metadata.get("date"):
            source_info += f" [Tanggal: {metadata['date']}]"
        if metadata.get("sentiment"):
            source_info += f" [Sentimen: {metadata['sentiment']}]"

        context_parts.append(
            f"📄 Dokumen {i} {source_info}\n{doc.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    # Format riwayat percakapan jika ada
    history_text = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-6:]:  # Ambil 6 pesan terakhir agar tidak terlalu panjang
            role = "Pengguna" if msg["role"] == "user" else "Asisten"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)

    tanggal_sekarang = datetime.now().strftime("%d %B %Y, %H:%M WIB")

    prompt = f"""Kamu adalah UNSIKA Reputation Intelligence Assistant — asisten analisis reputasi cerdas milik Universitas Singaperbangsa Karawang (UNSIKA).

Tanggal & Waktu Sekarang: {tanggal_sekarang}

=== PANDUAN PERAN ===
- Kamu memiliki akses ke database berita, media sosial, dan dokumen internal UNSIKA
- Kamu membantu tim PR, manajemen, dan pemangku kepentingan UNSIKA memahami kondisi reputasi universitas
- Kamu berbicara secara profesional namun tetap ramah dan mudah dipahami
- Kamu HANYA menjawab berdasarkan data yang tersedia, tidak mengarang fakta

=== DATA KONTEKS YANG RELEVAN ===
{context_text}

{"=== RIWAYAT PERCAKAPAN ===" + chr(10) + history_text if history_text else ""}

=== PERTANYAAN PENGGUNA ===
{query}

=== INSTRUKSI MENJAWAB ===
Jawab pertanyaan di atas dengan memperhatikan hal berikut:

1. **Gunakan data konteks** sebagai sumber utama jawaban
2. **Struktur jawaban** dengan jelas menggunakan poin-poin jika diperlukan
3. **Sertakan analisis** tidak hanya fakta mentah — interpretasikan apa artinya bagi reputasi UNSIKA
4. **Jika data tidak cukup**, katakan secara jujur: "Berdasarkan data yang tersedia, informasi mengenai hal ini belum cukup lengkap."
5. **Jika pertanyaan di luar konteks UNSIKA**, arahkan kembali dengan sopan
6. **Akhiri dengan insight** atau rekomendasi singkat jika relevan

Format jawaban menggunakan markdown agar mudah dibaca."""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Chatbot Error: {str(e)}"