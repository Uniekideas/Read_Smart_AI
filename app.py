###############################################
# Read Smart AI - Minimal Fix Version
# HuggingFaceEmbeddings removed COMPLETELY
# Using Gemini Embeddings: text-embedding-004
###############################################

import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
import io
from datetime import datetime
from textwrap import dedent

# LangChain utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Optional exports
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except:
    HAS_DOCX = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except:
    HAS_FPDF = False


###############################
#   GOOGLE GENERATIVE AI INIT
###############################
def init_genai():
    key = st.secrets.get("GOOGLE_API_KEY") if st.secrets else None
    if not key:
        st.error("Missing GOOGLE_API_KEY in Streamlit Secrets.")
        return False
    genai.configure(api_key=key)
    return True


###############################
#   PDF → TEXT
###############################
def pdf_to_text(file):
    reader = PdfReader(file)
    texts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            texts.append({"page": i + 1, "text": t})
    return texts


###############################
#   BUILD CHUNKS
###############################
def build_chunks_from_docs(docs_texts, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, metadata = [], []

    for d in docs_texts:
        content = d["text"]
        splits = splitter.split_text(content)

        for idx, s in enumerate(splits):
            chunks.append(s)
            metadata.append({
                "source_name": d.get("name", "uploaded_pdf"),
                "page": d.get("page"),
                "chunk_index": idx
            })

    return chunks, metadata


###############################
#   EMBEDDINGS (GEMINI)
###############################
def embed_texts(texts):
    """Embeds a list of texts using Gemini Embedding model."""
    try:
        model = "models/text-embedding-004"
        result = genai.embed_content(model=model, content=texts)
        return result["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {e}")
        raise e


###############################
#   BUILD VECTORSTORE (CHROMA)
###############################
def build_vectorstore(chunks, metadata):
    if not init_genai():
        return None

    embeddings = []
    batch_size = 20

    # Gemini allows batching (recommended)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        resp = genai.embed_content(
            model="models/text-embedding-004",
            content=batch
        )
        vecs = resp["embedding"]
        embeddings.extend(vecs)

    chroma_db = Chroma.from_embeddings(
        embeddings=embeddings,
        texts=chunks,
        metadatas=metadata
    )
    return chroma_db


###############################
#   RETRIEVE DOCS
###############################
def retrieve_top_docs(vectorstore, query, k=4):
    resp = genai.embed_content(model="models/text-embedding-004", content=query)
    query_vec = resp["embedding"]
    docs = vectorstore.similarity_search_by_vector(query_vec, k=k)
    return docs


###############################
#   GEMINI LLM CALL
###############################
def call_gemini_with_context(prompt):
    try:
        resp = genai.generate(model="gemini-1.5-flash", prompt=prompt)
        if hasattr(resp, "candidates"):
            return resp.candidates[0].content[0].text
        return str(resp)
    except:
        # fallback API
        resp = genai.generate(model="gemini-1.5-flash", prompt=prompt)
        return str(resp)


###############################################
#   STREAMLIT SESSION STATE INIT
###############################################
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "metadatas" not in st.session_state:
    st.session_state.metadatas = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


###############################################
#   UI SETUP
###############################################
st.set_page_config(layout="wide", page_title="Read Smart AI")
st.title("Read Smart AI — Dashboard & Chat")


###############################
# SIDEBAR
###############################
with st.sidebar:
    st.header("Controls")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    st.subheader("Embedding Settings")
    chunk_size = st.number_input("Chunk size", 256, 4000, 1000, 128)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 200, 50)
    top_k = st.slider("Top K", 1, 8, 4)

    st.subheader("Actions")
    build_index_btn = st.button("Build / Rebuild Index")
    summarize_btn = st.button("Summarize All Docs")
    clear_btn = st.button("Clear All")

    export_md_btn = st.button("Export Chat (Markdown)")
    export_docx_btn = st.button("Export Chat (Docx)")
    export_pdf_btn = st.button("Export Chat (PDF)")


###############################################
# DOCUMENT LOADING
###############################################
if uploaded_files:
    pages = []
    for f in uploaded_files:
        try:
            extracted = pdf_to_text(f)
            for p in extracted:
                pages.append({"name": f.name, "page": p["page"], "text": p["text"]})
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    st.session_state.uploaded_docs = pages
    st.success(f"Loaded {len(pages)} pages.")


###############################################
# CLEAR ALL
###############################################
if clear_btn:
    st.session_state.vector_store = None
    st.session_state.chunks = []
    st.session_state.metadatas = []
    st.session_state.chat_history = []
    st.session_state.uploaded_docs = []
    st.success("Cleared all.")


###############################################
# BUILD INDEX
###############################################
if build_index_btn:
    if not st.session_state.uploaded_docs:
        st.warning("Upload PDFs first.")
    else:
        with st.spinner("Building index…"):
            chunks, metas = build_chunks_from_docs(
                st.session_state.uploaded_docs,
                chunk_size,
                chunk_overlap
            )
            st.session_state.chunks = chunks
            st.session_state.metadatas = metas

            st.session_state.vector_store = build_vectorstore(chunks, metas)

        st.success(f"Indexed {len(chunks)} chunks.")


###############################################
# MAIN LAYOUT — CHAT
###############################################
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Conversation")

    # Display chat history
    for msg in st.session_state.chat_history:
        bg = "#eef3ff" if msg["role"] == "user" else "#f7f7f7"
        st.markdown(
            f"<div style='background:{bg}; padding:10px; border-radius:8px; margin:6px 0'>{msg['text']}</div>",
            unsafe_allow_html=True
        )

    query = st.text_input("Ask a question…")
    ask_btn = st.button("Ask")


###############################################
# RIGHT COLUMN (DOCS)
###############################################
with col2:
    st.subheader("Uploaded Documents")
    if st.session_state.uploaded_docs:
        for d in st.session_state.uploaded_docs:
            st.write(f"- **{d['name']}** (page {d['page']})")
    else:
        st.info("No documents loaded.")


###############################################
# QUESTION HANDLER
###############################################
if ask_btn:
    if not query.strip():
        st.warning("Enter a question.")
    elif not st.session_state.vector_store:
        st.warning("Build the index first.")
    else:
        docs = retrieve_top_docs(st.session_state.vector_store, query, top_k)

        taken = []
        for d in docs:
            meta = d.metadata
            taken.append({
                "source_name": meta["source_name"],
                "page": meta["page"],
                "chunk_index": meta["chunk_index"],
                "text": d.page_content
            })

        context = "\n\n---\n\n".join([
            f"Source: {t['source_name']} (page {t['page']})\n\n{t['text']}"
            for t in taken
        ])

        prompt = dedent(f"""
        Answer the question using ONLY the context below.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """)

        if not init_genai():
            st.stop()

        with st.spinner("Thinking…"):
            answer = call_gemini_with_context(prompt)

        st.session_state.chat_history.append({"role": "user", "text": query})
        st.session_state.chat_history.append({"role": "assistant", "text": answer})

        st.experimental_rerun()
