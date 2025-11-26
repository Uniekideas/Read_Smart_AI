import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="📘 Read Smart AI", layout="wide")
st.title("📘 Read Smart AI")

CHROMA_DIR = "chroma_store"  # Works on Streamlit Cloud


# ============================================================
# GOOGLE GENAI API KEY
# ============================================================

def init_genai():
    key = st.secrets.get("GEMINI_API_KEY")
    if not key:
        st.error("❌ Missing GEMINI_API_KEY in Streamlit Secrets.")
        return False
    genai.configure(api_key=key)
    return True


# ============================================================
# PDF READER
# ============================================================

def extract_pdf_text(files):
    text = ""
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# ============================================================
# TEXT SPLITTING
# ============================================================

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    chunks = splitter.split_text(text)
    metadata = [{"source": f"chunk_{i}"} for i in range(len(chunks))]
    return chunks, metadata


# ============================================================
# BUILD VECTORSTORE
# ============================================================

def build_vectorstore(chunks, metadata):
    if not init_genai():
        return None

    embeddings = []
    batch_size = 20

    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            resp = genai.embed_content(
                model="models/text-embedding-004",
                content=batch
            )
            embeddings.extend(resp["embedding"])
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

    vectorstore = Chroma(
        collection_name="readsmart_collection",
        embedding_function=None,
        persist_directory=CHROMA_DIR
    )

    vectorstore.add_embeddings(
        embeddings=embeddings,
        metadatas=metadata,
        documents=chunks
    )

    vectorstore.persist()
    return vectorstore


# ============================================================
# LOAD VECTORSTORE
# ============================================================

def load_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        return None
    try:
        return Chroma(
            collection_name="readsmart_collection",
            embedding_function=None,
            persist_directory=CHROMA_DIR
        )
    except:
        return None


# ============================================================
# RAG QUERY
# ============================================================

def answer_question(vectorstore, query):
    if not init_genai():
        return "API key missing."

    q_emb = genai.embed_content(
        model="models/text-embedding-004",
        content=query
    )["embedding"]

    docs = vectorstore.similarity_search_by_vector(q_emb, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use ONLY the context below to answer:

CONTEXT:
{context}

QUESTION:
{query}
"""

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


# ============================================================
# SUMMARY
# ============================================================

def summarize_text(text):
    if not init_genai():
        return "Missing API key."

    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Summarize this in clear bullet points:\n\n{text}"
    return model.generate_content(prompt).text


# ============================================================
# UI
# ============================================================

st.sidebar.header("📁 Upload PDFs")

pdf_files = st.sidebar.file_uploader(
    "Upload one or multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("📌 Build Index"):
    if not pdf_files:
        st.sidebar.error("Upload PDFs first.")
    else:
        with st.spinner("Extracting text..."):
            full_text = extract_pdf_text(pdf_files)

        with st.spinner("Splitting text..."):
            chunks, metadata = split_text(full_text)

        with st.spinner("Creating embeddings..."):
            store = build_vectorstore(chunks, metadata)

        if store:
            st.sidebar.success("✔ Vectorstore built!")
        else:
            st.sidebar.error("Failed to build vectorstore.")

vectorstore = load_vectorstore()

# QUESTIONS
st.subheader("💬 Ask Questions")

if vectorstore:
    user_q = st.text_input("Ask something:")
    if user_q:
        with st.spinner("Thinking..."):
            answer = answer_question(vectorstore, user_q)
        st.write("### 🧠 Answer")
        st.write(answer)
else:
    st.info("⚠ Build the index first in the sidebar.")

# SUMMARY
st.subheader("📝 Generate Summary")

if pdf_files and st.button("Summarize PDF(s)"):
    full_text = extract_pdf_text(pdf_files)
    with st.spinner("Summarizing..."):
        summary = summarize_text(full_text)
    st.write("### Summary")
    st.write(summary)
