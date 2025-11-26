###############################################
# READ SMART AI — FINAL WORKING VERSION (2025)
# Gemini embeddings + Chroma persistent vectorstore
# Valid model names + model‑list printing for debugging
###############################################

import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
from datetime import datetime
from textwrap import dedent

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Optional exports
try:
    from docx import Document
    HAS_DOCX = True
except:
    HAS_DOCX = False

try:
    from fpdf import FPDF
    HAS_PDF = True
except:
    HAS_PDF = False

# Constants
CHROMA_DIR = "./chroma_store"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # valid as at late 2025

# ------------------------------------------------
# Initialize Gemini / GenAI
# ------------------------------------------------
def init_genai():
    key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        st.error("🚫 Missing GOOGLE_API_KEY. Add it to Streamlit secrets or environment variables.")
        return False
    genai.configure(api_key=key)
    return True

# ------------------------------------------------
# Fetch available Gemini models (for debugging / selection)
# ------------------------------------------------
def get_available_models():
    try:
        models = genai.list_models()
        # models is assumed list of dicts or strings
        names = []
        for m in models:
            if isinstance(m, dict) and "id" in m:
                names.append(m["id"])
            elif isinstance(m, str):
                names.append(m)
        return names
    except Exception as e:
        st.warning(f"Could not fetch model list: {e}")
        return []

# ------------------------------------------------
# PDF → text
# ------------------------------------------------
def pdf_to_text(file):
    reader = PdfReader(file)
    pages = []
    for i, p in enumerate(reader.pages):
        text = p.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

# ------------------------------------------------
# Chunk documents
# ------------------------------------------------
def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, metadata = [], []
    for d in docs:
        splits = splitter.split_text(d["text"])
        for idx, s in enumerate(splits):
            chunks.append(s)
            metadata.append({
                "source": d.get("name", "uploaded_pdf"),
                "page": d.get("page"),
                "chunk": idx
            })
    return chunks, metadata

# ------------------------------------------------
# Embedding with Gemini
# ------------------------------------------------
def embed_batch(texts):
    if not init_genai():
        raise RuntimeError("Gemini not configured")
    resp = genai.embed_content(model="models/text-embedding-004", content=texts)
    return resp["embedding"]

# ------------------------------------------------
# Build / Persist Chroma Vectorstore
# ------------------------------------------------
def build_vectorstore(chunks, metadata):
    if not init_genai():
        return None
    os.makedirs(CHROMA_DIR, exist_ok=True)

    store = Chroma(collection_name="readsmart", embedding_function=None, persist_directory=CHROMA_DIR)
    try:
        store.delete_collection()
    except Exception:
        pass
    store = Chroma(collection_name="readsmart", embedding_function=None, persist_directory=CHROMA_DIR)

    embeddings = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        emb = embed_batch(batch)
        embeddings.extend(emb)

    store._collection.add(
        ids=[f"id_{i}" for i in range(len(embeddings))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadata
    )

    store.persist()
    return store

# ------------------------------------------------
# Retrieve via similarity search
# ------------------------------------------------
def retrieve(store, query, k=4):
    q_emb = embed_batch([query])[0]
    results = store._collection.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({"text": doc, "meta": meta})
    return docs

# ------------------------------------------------
# Gemini LLM call — safe version
# ------------------------------------------------
def ask_gemini(prompt_text, model_name: str):
    if not init_genai():
        return "Error: missing API key."
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"Error: cannot load model '{model_name}': {e}"

    try:
        resp = model.generate_content(prompt_text)
    except Exception as e:
        return f"Error during generation: {e}\n\nAvailable models: {get_available_models()}"

    # Try to extract text
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict):
        if "candidates" in resp and resp["candidates"]:
            cand = resp["candidates"][0]
            # often nested content structure
            if isinstance(cand, dict):
                if "content" in cand and isinstance(cand["content"], list):
                    first = cand["content"][0]
                    if isinstance(first, dict) and "text" in first:
                        return first["text"]
                if "text" in cand:
                    return cand["text"]
        if "text" in resp:
            return resp["text"]
    return str(resp)

# ------------------------------------------------
# Streamlit state initialization
# ------------------------------------------------
for key, default in {"docs": [], "chunks": [], "meta": [], "store": None, "chat": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------
# UI Setup
# ------------------------------------------------
st.set_page_config(page_title="Read Smart AI", layout="wide")
st.title("📘 Uniek Ideas - Read Smart AI")

with st.sidebar:
    st.header("Upload PDF(s)")
    uploaded_files = st.file_uploader("Select PDF file(s)", type="pdf", accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Model & Retrieval Settings")

    # Show available models (for user visibility)
    models = get_available_models()
    if models:
        st.markdown("**Available Gemini models:**")
        for m in models:
            st.write(f"- {m}")
    else:
        st.markdown("No models fetched — check your API key.")

    gemini_model = st.selectbox(
        "Choose Gemini model",
        options=models or [DEFAULT_GEMINI_MODEL],
        index=0
    )

    cs = st.number_input("Chunk size", 256, 4000, 1000, 128)
    co = st.number_input("Chunk overlap", 0, 1000, 200, 50)
    topk = st.slider("Top-K (retrieval)", 1, 10, 4)

    st.markdown("---")
    st.subheader("Actions")
    btn_build = st.button("Build / Rebuild Index")
    btn_clear = st.button("Clear All")

# ------------------------------------------------
# Load PDF → docs
# ------------------------------------------------
if uploaded_files:
    pages = []
    for f in uploaded_files:
        extracted = pdf_to_text(f)
        for p in extracted:
            pages.append({"name": f.name, "page": p["page"], "text": p["text"]})
    if pages:
        st.session_state.docs = pages
        st.success(f"Loaded {len(pages)} pages from {len(uploaded_files)} file(s).")

# ------------------------------------------------
# Clear state
# ------------------------------------------------
if btn_clear:
    st.session_state.docs = []
    st.session_state.store = None
    st.session_state.chunks = []
    st.session_state.meta = []
    st.session_state.chat = []
    st.success("Cleared all in-memory state. (Persistent DB remains.)")

# ------------------------------------------------
# Build Index
# ------------------------------------------------
if btn_build:
    if not st.session_state.docs:
        st.warning("Please upload PDF(s) first.")
    else:
        with st.spinner("Indexing..."):
            chunks, meta = chunk_documents(st.session_state.docs, cs, co)
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.store = build_vectorstore(chunks, meta)
        st.success(f"Indexed {len(chunks)} chunks into Chroma store.")

# ------------------------------------------------
# Main Chat UI
# ------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Chat & Q&A")
    for msg in st.session_state.chat:
        bg = "#e8f0ff" if msg.get("role") == "user" else "#f6f6f6"
        st.markdown(
            f"<div style='background:{bg}; padding:10px; border-radius:8px; margin:5px 0'>{msg.get('text')}</div>",
            unsafe_allow_html=True
        )

    query = st.text_input("Ask something about the uploaded PDFs")
    ask = st.button("Ask")

with col2:
    st.subheader("Uploaded Documents")
    if st.session_state.docs:
        for d in st.session_state.docs:
            st.write(f"- **{d['name']}** (page {d['page']})")
    else:
        st.info("No documents uploaded.")

# ------------------------------------------------
# Handle Question
# ------------------------------------------------
if ask:
    if not query or not query.strip():
        st.warning("Please type a question.")
    elif not st.session_state.store:
        st.warning("Please build the index first (Use sidebar).")
    else:
        docs = retrieve(st.session_state.store, query, topk)
        context = "\n\n---\n\n".join([
            f"Source: {d['meta']['source']} (page {d['meta']['page']})\n{d['text']}"
            for d in docs
        ])
        prompt = dedent(f"""
        Answer the question below using ONLY the context.  
        If the answer is not in the context, say "I don't know."

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """)

        with st.spinner("Thinking..."):
            answer = ask_gemini(prompt, model_name=gemini_model)

        st.session_state.chat.append({"role": "user", "text": query})
        st.session_state.chat.append({"role": "assistant", "text": answer})
        st.experimental_rerun()
