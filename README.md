# 🧠📄 PDF Chat Assistant

**A Hybrid Retrieval-Augmented Generation (RAG) System powered by Google Gemini**

This project enables chatting with multiple PDF documents using AI. The system combines **semantic search** and **keyword search** to retrieve relevant context and generate responses — grounded strictly in the content of uploaded PDFs.

---

## 🚀 Features

- 📂 **Upload and process multiple PDFs**
- 🔍 Hybrid search: **FAISS (dense)** + **BM25 (lexical)**
- 🧠 Powered by **Gemini AI Models**
- 🎯 **Zero hallucination** – answers only from PDFs
- 🔄 Dynamic knowledge base building
- 📌 Displays source PDFs for citation transparency
- 🎨 Clean UI built with Streamlit
- 🔧 **No model training or fine-tuning** required

---

## 🛠️ Tech Stack

| Component          | Purpose                             |
|--------------------|-------------------------------------|
| Python             | Core development language           |
| Streamlit          | Web UI framework                    |
| PyPDF              | PDF text extraction                 |
| NumPy              | Numerical operations                |
| FAISS              | Dense vector semantic search        |
| Rank-BM25          | Keyword-based document ranking      |
| Google Gemini API  | Embeddings + Large Language Models  |

---

## 🧪 AI Models Used

- **`text-embedding-004`** — Generates vector embeddings for PDF chunks and user queries  
- **`gemini-2.5-pro`** — Main LLM used for accurate, grounded answers  
- **`gemini-2.5-flash`** — Faster model for low-latency responses  

Also supports:
- `gemini-flash-latest`
- `gemini-pro-latest`

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

# Create a Python virtual environment
python -m venv ragenv
source ragenv/bin/activate  # (Windows: ragenv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
