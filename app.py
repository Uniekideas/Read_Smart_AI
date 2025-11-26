import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ============================================================
# CONFIG & SETUP
# ============================================================

st.set_page_config(page_title="📘 Read Smart AI", layout="wide")
st.title("📘 Read Smart AI")

CHROMA_DIR = "./chroma_store"

# ============================================================
# GOOGLE GEMINI KEY
# ============================================================

def init_genai():
    key = st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        st.error("❌ GEMINI_API_KEY missing. Add it in Streamlit Secrets.")
        return None
    genai.configure(api_key=key)
    return True


# ============================================================
# PDF LOADING
# ============================================================

def extract_pdf_text(files):
    all_text = ""
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            all_text += page.extract_text() or ""
    return all_text


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
# BUILD VECTORSTORE (PERSISTENT)
# ============================================================

def build_vectorstore(chunks, metadata):
    if not init_genai():
        return None

    embeddings = []
    batch_size = 20

    # Generate embeddings in batches
    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            resp = genai.embed_content(
                model="models/text-embedding-004",
                content=batch
            )
            embeddings.extend(resp["embedding"])
    except Exception as e:
        st.error(f"Embedding Error: {str(e)}")
        return None

    # Build persistent Chroma DB
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
# LOAD EXISTING VECTORSTORE
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

    # Embed query
    q_emb = genai.embed_content(
        model="models/text-embedding-004",
        content=query
    )["embedding"]

    # Search in vectorstore
    docs = vectorstore.similarity_search_by_vector(q_emb, k=5)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant. Answer the user question using ONLY the context below:

CONTEXT:
{context}

QUESTION:
{query}

Provide a clear and concise answer.
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


# ============================================================
# PDF SUMMARY
# ============================================================

def summarize_text(text):
    if not init_genai():
        return "Missing API key."

    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Summarize the following document in clear bullet points:\n\n{text}"
    return model.generate_content(prompt).text


# ============================================================
# STREAMLIT UI
# ============================================================

# Sidebar
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

        with st.spinner("Generating embeddings & saving vectorstore..."):
            store = build_vectorstore(chunks, metadata)

        if store:
            st.sidebar.success("✅ Vectorstore built & saved!")
        else:
            st.sidebar.error("❌ Failed to build vectorstore.")

# Load vectorstore (if exists)
vectorstore = load_vectorstore()

# MAIN AREA
st.subheader("💬 Ask Questions")

if vectorstore:
    user_query = st.text_input("Ask something about your PDFs:")
    if user_query:
        with st.spinner("Thinking..."):
            answer = answer_question(vectorstore, user_query)
        st.write("### 🧠 Answer")
        st.write(answer)
else:
    st.info("⚠ Build the index first in the sidebar.")

# PDF SUMMARY SECTION
st.subheader("📝 Generate PDF Summary")

if pdf_files and st.button("Summarize PDF(s)"):
    full_text = extract_pdf_text(pdf_files)
    with st.spinner("Summarizing..."):
        summary = summarize_text(full_text)
    st.write("### Summary")
    st.write(summary)
