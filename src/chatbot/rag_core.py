import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# GUNAKAN LIBRARY YANG SESUAI DENGAN REQUIREMENTS.TXT
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project = os.path.abspath(os.path.join(current_dir, '../..'))
if root_project not in sys.path:
    sys.path.append(root_project)

load_dotenv()

# Konfigurasi Gemini (Versi Standar)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

_VECTORSTORE = None


def get_vector_db():
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    save_path = os.path.join(root_project, "data", "faiss_index")
    _VECTORSTORE = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    return _VECTORSTORE


def ask_bot(query: str):
    vectorstore = get_vector_db()
    docs = vectorstore.similarity_search(query, k=15)
    context_text = "\n\n".join([f"- {doc.page_content}" for doc in docs])

    prompt = f"""Anda adalah Sistem Analis Reputasi UNSIKA. Gunakan data berikut untuk menjawab:
    {context_text}
    Pertanyaan: {query}"""

    # Eksekusi dengan library yang benar
    response = model.generate_content(prompt)
    return response.text