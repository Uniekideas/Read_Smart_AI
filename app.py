###############################################
# READ SMART AI — FULLY WORKING VERSION 2025
# Gemini embeddings + Chroma persistent vectorstore
# Streamlit Cloud compatible
###############################################

import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
from datetime import datetime
from textwrap import dedent

# LangChain imports for text splitting + Chroma wrapper
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

# -------------------------
# Gemini Initialization
# -------------------------
def init_genai():
    key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        st.error("Missing GOOGLE_API_KEY (set in Streamlit secrets or env)")
        return False
    genai.configure(api_key=key)
    return True

# -------------------------
# PDF → Text
# -------------------------
def pdf_to_text(file):
    pages = []
    reader = PdfReader(file)
    for i, p in enumerate(reader.pages):
        text = p.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

# -------------------------
# Chunking
# -------------------------
def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, metadata = [], []
    for d in docs:
        splits = splitter.split_text(d["text"])
        for idx, s in enumerate(splits):
            chunks.append(s)
            metadata.append({
                "source": d["name"],
                "page": d["page"],
                "chunk": idx
            })
    return chunks, metadata

# -------------------------
# Gemini Embedding (batch)
# -------------------------
def embed_batch(texts):
    if not init_genai():
        raise RuntimeError("Gemini not configured")
    resp = genai.embed_content(model="models/text-embedding-004", content=texts)
    return resp["embedding"]

# -------------------------
# Vectorstore (Chroma)
# -------------------------
CHROMA_DIR = "./chroma_store"

def build_vectorstore(chunks, metadata):
    if not init_genai():
        return None

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Create store without embedding function
    store = Chroma(collection_name="readsmart", embedding_function=None, persist_directory=CHROMA_DIR)

    # Clear old collection if exists
    try:
        store.delete_collection()
    except:
        pass
    store = Chroma(collection_name="readsmart", embedding_function=None, persist_directory=CHROMA_DIR)

    # Embed in batches
    embeddings = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        emb = embed_batch(batch)
        embeddings.extend(emb)

    # Add directly to collection
    store._collection.add(
        ids=[f"id_{i}" for i in range(len(embeddings))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadata
    )

    store.persist()
    return store

# -------------------------
# Retrieval
# -------------------------
def retrieve(store, query, k=4):
    q_emb = embed_batch([query])[0]
    results = store._collection.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({"text": doc, "meta": meta})
    return docs

# -------------------------
# Gemini LLM (fixed)
# -------------------------
def ask_gemini(prompt_text, model_name="gemini-1.5-flash"):
    if not init_genai():
        return "Error: missing API key."
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt_text)
        if hasattr(response, "text"):
            return response.text
        if isinstance(response, dict):
            if "candidates" in response and response["candidates"]:
                cand = response["candidates"][0]
                if isinstance(cand, dict) and "content" in cand and isinstance(cand["content"], list):
                    first = cand["content"][0]
                    if isinstance(first, dict) and "text" in first:
                        return first["text"]
                if isinstance(cand, dict) and "text" in cand:
                    return cand["text"]
            if "text" in response:
                return response["text"]
        return str(response)
    except Exception as e:
        st.error(f"Gemini generation failed: {e}")
        return f"Error: {e}"

# -------------------------
# Streamlit state init
# -------------------------
for key, default in {"docs": [], "chunks": [], "meta": [], "store": None, "chat": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Read Smart AI", layout="wide")
st.title("📘 Read Smart AI — Stable Version")

with st.sidebar:
    st.header("Upload PDFs")
    files = st.file_uploader("Upload", type="pdf", accept_multiple_files=True)

    st.subheader("Settings")
    cs = st.number_input("Chunk size", 256, 4000, 1000, 128)
    co = st.number_input("Chunk overlap", 0, 1000, 200, 50)
    topk = st.slider("Top-K", 1, 10, 4)

    st.subheader("Actions")
    btn_build = st.button("Build Index")
    btn_clear = st.button("Clear All")

# -------------------------
# Load PDFs
# -------------------------
if files:
    pages = []
    for f in files:
        extracted = pdf_to_text(f)
        for p in extracted:
            pages.append({"name": f.name, "page": p["page"], "text": p["text"]})
    st.session_state.docs = pages
    st.success(f"Loaded {len(pages)} pages.")

# -------------------------
# Clear all
# -------------------------
if btn_clear:
    st.session_state.docs = []
    st.session_state.store = None
    st.session_state.chunks = []
    st.session_state.meta = []
    st.session_state.chat = []
    st.success("Cleared.")

# -------------------------
# Build index
# -------------------------
if btn_build:
    if not st.session_state.docs:
        st.warning("Upload PDFs first.")
    else:
        with st.spinner("Indexing..."):
            chunks, meta = chunk_documents(st.session_state.docs, cs, co)
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.store = build_vectorstore(chunks, meta)
        st.success(f"Indexed {len(chunks)} chunks.")

# -------------------------
# Main chat UI
# -------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Chat")
    for c in st.session_state.chat:
        bg = "#e8f0ff" if c["role"] == "user" else "#f6f6f6"
        st.markdown(f"<div style='background:{bg}; padding:10px; border-radius:8px; margin:5px 0'>{c['text']}</div>", unsafe_allow_html=True)

    query = st.text_input("Ask anything about your PDFs:")
    ask_btn = st.button("Ask")

with col2:
    st.subheader("Uploaded Pages")
    if st.session_state.docs:
        for d in st.session_state.docs:
            st.write(f"- **{d['name']}** (p{d['page']})")
    else:
        st.info("No documents.")

# -------------------------
# Handle question
# -------------------------
if ask_btn:
    if not query.strip():
        st.warning("Enter a question.")
    elif not st.session_state.store:
        st.warning("Build index first.")
    else:
        docs = retrieve(st.session_state.store, query, topk)
        context = "\n\n---\n\n".join([f"Source: {d['meta']['source']} (page {d['meta']['page']})\n{d['text']}" for d in docs])
        prompt = dedent(f"""
        Answer the following question using ONLY the context.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """)
        with st.spinner("Thinking..."):
            answer = ask_gemini(prompt)
        st.session_state.chat.append({"role": "user", "text": query})
        st.session_state.chat.append({"role": "assistant", "text": answer})
        st.experimental_rerun()
