###############################################
# READ SMART AI — FINAL STABLE STREAMLIT CLOUD
# Gemini embeddings + Chroma vectorstore
# Hardcoded model to avoid Cloud metadata errors
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
GEMINI_MODEL = "gemini-2.5-flash"  # hardcoded for Streamlit Cloud

# ------------------------------------------------
# Initialize Gemini
# ------------------------------------------------
def init_genai():
    key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        st.error("🚫 Missing GOOGLE_API_KEY. Add it to Streamlit secrets or environment variables.")
        return False
    genai.configure(api_key=key)
    return True

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
# Gemini embedding batch
# ------------------------------------------------
def embed_batch(texts):
    if not init_genai():
        raise RuntimeError("Gemini not configured")
    resp = genai.embed_content(model="models/text-embedding-004", content=texts)
    return resp["embedding"]

# ------------------------------------------------
# Build Chroma vectorstore
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
# Retrieve top-K
# ------------------------------------------------
def retrieve(store, query, k=4):
    q_emb = embed_batch([query])[0]
    results = store._collection.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({"text": doc, "meta": meta})
    return docs

# ------------------------------------------------
# Ask Gemini LLM
# ------------------------------------------------
def ask_gemini(prompt_text):
    if not init_genai():
        return "Error: missing API key."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt_text)
    except Exception as e:
        return f"Error during generation: {e}"

    # Extract text
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict):
        if "candidates" in resp and resp["candidates"]:
            cand = resp["candidates"][0]
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
# Streamlit session state
# ------------------------------------------------
for key, default in {"docs": [], "chunks": [], "meta": [], "store": None, "chat": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------
# UI
# ------------------------------------------------
st.set_page_config(page_title="Read Smart AI", layout="wide")
st.title("📘 Uniek Ideas — Read Smart AI")

with st.sidebar:
    st.header("Upload PDF(s)")
    uploaded_files = st.file_uploader("Select PDF file(s)", type="pdf", accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Settings")
    cs = st.number_input("Chunk size", 256, 4000, 1000, 128)
    co = st.number_input("Chunk overlap", 0, 1000, 200, 50)
    topk = st.slider("Top-K (retrieval)", 1, 10, 4)

    st.markdown("---")
    st.subheader("Actions")
    btn_build = st.button("Build / Rebuild Index")
    btn_clear = st.button("Clear All")

# ------------------------------------------------
# Load PDFs
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
            answer = ask_gemini(prompt)

        st.session_state.chat.append({"role": "user", "text": query})
        st.session_state.chat.append({"role": "assistant", "text": answer})
        st.rerun()  # FIXED: Changed from st.experimental_rerun()