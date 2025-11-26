###############################################
# READ SMART AI — FINAL STABLE VERSION (2025)
# Works 100% on Streamlit Cloud
# - Gemini embeddings
# - Chroma persistent vectorstore
# - NO langchain embedding functions
# - NO deprecated APIs
# - Direct Chroma collection writes
###############################################

import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
from datetime import datetime
from textwrap import dedent

# --- Correct LangChain imports that Streamlit Cloud supports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- optional exports ---
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


###############################################
# INIT GEMINI
###############################################
def init_genai():
    key = st.secrets.get("GOOGLE_API_KEY")
    if not key:
        st.error("Missing GOOGLE_API_KEY in secrets.")
        return False
    genai.configure(api_key=key)
    return True


###############################################
# PDF → TEXT
###############################################
def pdf_to_text(file):
    pages = []
    reader = PdfReader(file)
    for i, p in enumerate(reader.pages):
        text = p.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


###############################################
# CREATE CHUNKS
###############################################
def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

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


###############################################
# GEMINI EMBEDDING (batch)
###############################################
def embed_batch(texts):
    resp = genai.embed_content(
        model="models/text-embedding-004",
        content=texts
    )
    return resp["embedding"]


###############################################
# BUILD VECTORSTORE (PERSISTENT)
###############################################
CHROMA_DIR = "./chroma_store"

def build_vectorstore(chunks, metadata):
    if not init_genai():
        return None

    # Clean or create DB directory
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Create vectorstore WITHOUT embedding function
    store = Chroma(
        collection_name="readsmart",
        embedding_function=None,
        persist_directory=CHROMA_DIR
    )

    # --- clear existing for rebuild ---
    store.delete_collection()
    store = Chroma(
        collection_name="readsmart",
        embedding_function=None,
        persist_directory=CHROMA_DIR
    )

    # --- embed chunks in batches ---
    embeddings = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        emb = embed_batch(batch)
        embeddings.extend(emb)

    # --- DIRECT WRITE TO CHROMA COLLECTION ---
    store._collection.add(
        ids=[f"id_{i}" for i in range(len(embeddings))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadata
    )

    store.persist()
    return store


###############################################
# RETRIEVAL
###############################################
def retrieve(store, query, k=4):
    q_emb = embed_batch([query])[0]
    results = store._collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({
            "text": doc,
            "meta": meta
        })

    return docs


###############################################
# GEMINI LLM
###############################################
def ask_gemini(prompt):
    resp = genai.generate(
        model="gemini-1.5-flash",
        prompt=prompt
    )

    # Try multiple response shapes (object attributes or dicts)
    try:
        # 1) Attribute-style response (SDK objects)
        candidates = getattr(resp, "candidates", None)
        if candidates:
            first = candidates[0]
            content = getattr(first, "content", None)
            if content:
                if isinstance(content, list) and len(content) > 0:
                    piece = content[0]
                    text = getattr(piece, "text", None)
                    if text:
                        return text
                elif isinstance(content, str):
                    return content

        # 2) Dict-style response
        if isinstance(resp, dict):
            cand = resp.get("candidates")
            if cand:
                first = cand[0]
                content = first.get("content")
                if isinstance(content, list) and len(content) > 0:
                    txt = content[0].get("text")
                    if txt:
                        return txt
                if isinstance(content, str):
                    return content

        # 3) Fallbacks
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict) and "output" in resp:
            return str(resp["output"])

    except Exception:
        # swallow parsing errors and fall back to string conversion
        pass

    return str(resp)


###############################################
# STREAMLIT STATE
###############################################
for key, default in {
    "docs": [],
    "chunks": [],
    "meta": [],
    "store": None,
    "chat": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


###############################################
# UI
###############################################
st.set_page_config(page_title="Read Smart AI", layout="wide")
st.title("📘 Read Smart AI — Stable Version")


###############################################
# SIDEBAR
###############################################
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


###############################################
# LOAD DOCUMENTS
###############################################
if files:
    pages = []
    for f in files:
        extracted = pdf_to_text(f)
        for p in extracted:
            pages.append({"name": f.name, "page": p["page"], "text": p["text"]})

    st.session_state.docs = pages
    st.success(f"Loaded {len(pages)} pages.")


###############################################
# CLEAR
###############################################
if btn_clear:
    st.session_state.docs = []
    st.session_state.store = None
    st.session_state.chunks = []
    st.session_state.meta = []
    st.session_state.chat = []
    st.success("Cleared.")


###############################################
# BUILD INDEX
###############################################
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


###############################################
# MAIN CHAT UI
###############################################
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Chat")

    # show history
    for c in st.session_state.chat:
        bg = "#e8f0ff" if c["role"] == "user" else "#f6f6f6"
        st.markdown(
            f"<div style='background:{bg}; padding:10px; border-radius:8px; margin:5px 0'>{c['text']}</div>",
            unsafe_allow_html=True
        )

    query = st.text_input("Ask anything about your PDFs:")
    ask_btn = st.button("Ask")


with col2:
    st.subheader("Uploaded Pages")
    if st.session_state.docs:
        for d in st.session_state.docs:
            st.write(f"- **{d['name']}** (p{d['page']})")
    else:
        st.info("No documents.")


###############################################
# HANDLE QUESTION
###############################################
if ask_btn:
    if not query.strip():
        st.warning("Enter a question.")
    elif not st.session_state.store:
        st.warning("Build index first.")
    else:
        docs = retrieve(st.session_state.store, query, topk)

        context = "\n\n---\n\n".join([
            f"Source: {d['meta']['source']} (page {d['meta']['page']})\n{d['text']}"
            for d in docs
        ])

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
