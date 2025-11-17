import streamlit as st
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import faiss
from typing import List
from config import get_api_key

# =========================
# Page Config & Global Styles
# =========================
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="🧠📄",
    layout="wide"
)

# Lighter, clean UI + readable text
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background: #f3f4f6;
        color: #020617;
    }

    .main {
        padding-top: 1rem;
    }

    .app-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #111827;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }

    .chat-bubble-user {
        background: #020617;
        padding: 0.7rem 0.9rem;
        border-radius: 999px;
        margin-bottom: 0.4rem;
        display: inline-block;
        max-width: 90%;
        color: #f9fafb;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .chat-bubble-bot {
        background: #111827;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        display: inline-block;
        max-width: 90%;
        color: #f9fafb;
        font-size: 1rem;
        line-height: 1.6;
    }

    .chat-label-user {
        font-size: 0.8rem;
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 0.15rem;
    }

    .chat-label-bot {
        font-size: 0.8rem;
        font-weight: 600;
        color: #4f46e5;
        margin-bottom: 0.15rem;
    }

    .stTextInput>div>div>input {
        background: #ffffff !important;
        border-radius: 999px !important;
        border: 1px solid #d1d5db !important;
        padding: 0.65rem 1.1rem !important;
        color: #111827 !important;
    }

    .stTextInput>label {
        display: none !important;
    }

    .send-button>button {
        border-radius: 999px !important;
        height: 42px !important;
        width: 42px !important;
        border: none !important;
        background: linear-gradient(135deg, #6366f1, #ec4899) !important;
        color: white !important;
        font-size: 1.2rem !important;
    }

    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid rgba(31,41,55,0.9);
    }

    .sidebar-card {
        background: #020617;
        border-radius: 16px;
        padding: 0.9rem;
        border: 1px solid rgba(75,85,99,0.9);
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: #f9fafb;
    }

    .sidebar-sub {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }

    .stButton>button {
        border-radius: 999px;
        border: none;
        padding: 0.4rem 0.9rem;
        font-weight: 500;
        background: linear-gradient(135deg, #4f46e5, #8b5cf6);
        color: #f9fafb;
    }

    .clear-btn>button {
        background: transparent !important;
        border-radius: 999px !important;
        border: 1px solid #f97373 !important;
        color: #fecaca !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Session State
# =========================
if "kb_built" not in st.session_state:
    st.session_state.kb_built = False
    st.session_state.chunks = None
    st.session_state.faiss_index = None
    st.session_state.bm25 = None
    st.session_state.chat_history = []  # list of {"question":..., "answer":...}
    st.session_state.chunk_sources = None  # parallel list: source filename for each chunk

# key used to reset the uploader
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =========================
# Helper Functions
# =========================
SYSTEM_PROMPT = (
    "You are an expert assistant that answers ONLY using the information from the provided PDF context.\n"
    "- If the answer is not clearly present in the context, say you cannot find it in the documents.\n"
    "- Do not invent information.\n"
    "- Be concise and clear. Use bullet points where helpful.\n"
)

def set_api_key():
    key = get_api_key()
    if not key:
        st.error("❌ No Gemini API Key found in config.py (empty string).")
        st.stop()
    if not key.startswith("AIza"):
        st.error("❌ Gemini API Key in config.py looks invalid (doesn't start with 'AIza').")
        st.stop()
    genai.configure(api_key=key)

def chunk_text(text, size=1000, overlap=200):
    """
    Chunk a single document's text into overlapping segments.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def read_and_chunk_pdfs(files):
    """
    Read multiple PDFs, extract text, and create chunks
    while tracking which file each chunk came from.

    Returns:
        all_chunks: list[str]
        all_sources: list[str]  (same length as all_chunks)
    """
    all_chunks = []
    all_sources = []

    for file in files:
        reader = PdfReader(file)
        name = getattr(file, "name", "Uploaded PDF")

        text = ""
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                text += extracted + "\n"

        if not text.strip():
            continue

        file_chunks = chunk_text(text)
        all_chunks.extend(file_chunks)
        all_sources.extend([name] * len(file_chunks))

    return all_chunks, all_sources

def embed_texts(texts: List[str]):
    vectors = []
    for c in texts:
        r = genai.embed_content(model="models/text-embedding-004", content=c)
        vectors.append(np.array(r["embedding"], dtype=np.float32))
    return np.vstack(vectors)

def build_hybrid_index(chunks):
    embeddings = embed_texts(chunks)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    bm25 = BM25Okapi([c.lower().split() for c in chunks])
    return index, bm25

def hybrid_search(query: str, top_k_dense: int = 8, top_k_bm25: int = 8, final_k: int = 5):
    q_vec = embed_texts([query])[0].reshape(1, -1)
    faiss.normalize_L2(q_vec)

    D, I = st.session_state.faiss_index.search(q_vec, top_k_dense)
    dense_hits = I[0].tolist()

    scores = st.session_state.bm25.get_scores(query.lower().split())
    bm25_hits = np.argsort(scores)[::-1][:top_k_bm25].tolist()

    k = 60
    dense_rank = {doc_id: rank for rank, doc_id in enumerate(dense_hits)}
    bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_hits)}

    fused_scores = {}
    for doc_id in set(dense_hits + bm25_hits):
        score = 0.0
        if doc_id in dense_rank:
            score += 1.0 / (k + dense_rank[doc_id] + 1)
        if doc_id in bm25_rank:
            score += 1.0 / (k + bm25_rank[doc_id] + 1)
        fused_scores[doc_id] = score

    ranked = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return ranked[:final_k]

def answer(query: str, model_name: str) -> str:
    if not st.session_state.kb_built:
        return "Knowledge base is not ready yet. Please upload PDFs and build it first."

    doc_ids = hybrid_search(query)
    context = "\n\n---\n\n".join(st.session_state.chunks[i] for i in doc_ids)

    # figure out which PDFs were used
    if st.session_state.chunk_sources:
        used_sources = sorted(set(st.session_state.chunk_sources[i] for i in doc_ids))
        sources_text = "\n".join(f"- {s}" for s in used_sources)
        sources_for_prompt = f"\n\nThe relevant information above was retrieved from these PDFs:\n{sources_text}\n"
    else:
        sources_for_prompt = ""

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context from PDFs:\n{context}\n"
        f"{sources_for_prompt}\n"
        f"User question: {query}\n\n"
        f"Answer using ONLY the context above:"
    )

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    answer_text = response.text

    # Add a small "Sources used" section at the end of the answer for the user
    if st.session_state.chunk_sources:
        used_sources = sorted(set(st.session_state.chunk_sources[i] for i in doc_ids))
        answer_text += "\n\n---\nSources used:\n" + "\n".join(f"- {s}" for s in used_sources)

    return answer_text

# =========================
# Sidebar Layout
# =========================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-card">'
        '<div class="sidebar-title">📂 Upload PDF</div>'
        '<div class="sidebar-sub">Choose one or more PDF files.</div>',
        unsafe_allow_html=True
    )

    # use dynamic key so we can "reset" uploader
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    build_kb = st.button("🔧 Build Knowledge Base", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    clear = st.container()
    with clear:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        clear_pressed = st.button("🧹 Clear PDFs & Chat", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🧠 Model")

    model_name = st.selectbox(
        "Choose Gemini model",
        [
            "models/gemini-2.5-pro",
            "models/gemini-2.5-flash",
            "models/gemini-flash-latest",
            "models/gemini-pro-latest",
        ],
        index=1
    )

# Sidebar actions
if clear_pressed:
    # 1) Clear knowledge base + chat + sources
    st.session_state.kb_built = False
    st.session_state.chunks = None
    st.session_state.faiss_index = None
    st.session_state.bm25 = None
    st.session_state.chat_history = []
    st.session_state.chunk_sources = None
    # 2) Reset uploader by changing its key
    st.session_state.uploader_key += 1
    st.success("Cleared PDFs and chat history. You can upload new files.")
    st.rerun()

if build_kb:
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("Building hybrid knowledge base from your PDFs..."):
            set_api_key()
            chunks, sources = read_and_chunk_pdfs(uploaded_files)
            if not chunks:
                st.error("No readable text found in uploaded PDFs.")
            else:
                faiss_index, bm25 = build_hybrid_index(chunks)

                st.session_state.chunks = chunks
                st.session_state.faiss_index = faiss_index
                st.session_state.bm25 = bm25
                st.session_state.chunk_sources = sources
                st.session_state.kb_built = True
                st.success("✅ Knowledge Base Ready! You can now ask questions.")

# =========================
# Main Layout (Title + Chat)
# =========================
left, right = st.columns([0.05, 0.95])

with right:
    st.markdown(
        """
        <div class="app-title"> 🧠📄PDF Chat Assistant</div>
        <div class="app-subtitle">💬 Ask questions across one or more PDFs and get answers grounded strictly in your documents.</div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.chat_history:
        st.info("Upload one or more PDFs on the left, click **Build Knowledge Base**, then ask any question about your documents here.")
    else:
        for turn in st.session_state.chat_history:
            st.markdown('<div class="chat-label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-user">{turn["question"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-label-bot">Assistant</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-bot">{turn["answer"]}</div>', unsafe_allow_html=True)

    # Bottom input bar
    with st.form(key="query_form", clear_on_submit=True):
        input_cols = st.columns([10, 1])
        with input_cols[0]:
            query = st.text_input(
                "",
                placeholder="Ask a question about your PDF(s)...",
                key="user_query",
            )
        with input_cols[1]:
            submitted = st.form_submit_button("➤", use_container_width=True)

    if submitted and query.strip():
        if not st.session_state.kb_built:
            st.warning("⚠️ Upload PDFs and click **Build Knowledge Base** first.")
        else:
            set_api_key()
            with st.spinner("Searching your PDFs & generating answer..."):
                result = answer(query.strip(), model_name)

            st.session_state.chat_history.append(
                {"question": query.strip(), "answer": result}
            )
            st.rerun()
