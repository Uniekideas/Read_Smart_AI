import streamlit as st
from PyPDF2 import PdfReader
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# -----------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Read Smart AI", layout="wide")

st.title("📘 Read Smart AI — Dashboard & Chat")


# -----------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "texts" not in st.session_state:
    st.session_state.texts = []
    st.session_state.metadatas = []


# -----------------------------------------------------
# EXTRACT TEXT FROM PDFs
# -----------------------------------------------------
def extract_pdf_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""


# -----------------------------------------------------
# SPLITTING INTO CHUNKS
# -----------------------------------------------------
def split_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)

    metadatas = [{"source": f"chunk{i}"} for i in range(len(chunks))]

    return chunks, metadatas


# -----------------------------------------------------
# BUILD VECTORSTORE
# -----------------------------------------------------
def build_vectorstore(texts, metadatas):
    embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vector_store


# -----------------------------------------------------
# ANSWER A QUESTION
# -----------------------------------------------------
def answer_question(query):
    if st.session_state.vector_store is None:
        st.warning("❗ Build the index first.")
        return ""

    docs = st.session_state.vector_store.similarity_search(query, k=3)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Construct simple prompt
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
You are Read Smart AI. Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""

    response = llm.invoke(prompt)

    return response.content


# -----------------------------------------------------
# SIDEBAR - PDF UPLOAD & SETTINGS
# -----------------------------------------------------
with st.sidebar:
    st.header("Controls")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        for file in uploaded:
            text = extract_pdf_text(file)
            st.session_state.documents.append((file.name, text))
        st.success(f"Loaded {len(uploaded)} PDF(s).")

    st.subheader("Embedding Settings")
    chunk_size = st.number_input("Chunk size", 200, 2000, 1000)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 200)

    if st.button("Build Index"):
        all_text = ""

        for name, doc_text in st.session_state.documents:
            all_text += doc_text + "\n"

        if len(all_text.strip()) < 10:
            st.error("Upload a valid PDF first.")
        else:
            with st.spinner("Building index..."):
                texts, metadatas = split_into_chunks(
                    all_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                st.session_state.texts = texts
                st.session_state.metadatas = metadatas

                st.session_state.vector_store = build_vectorstore(
                    texts,
                    metadatas
                )

            st.success("Index built successfully! 🎉")


# -----------------------------------------------------
# MAIN CHAT UI
# -----------------------------------------------------
st.subheader("Conversation")

query = st.text_input("Ask a question…")

if st.button("Ask"):
    if not query.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Thinking…"):
            answer = answer_question(query)
        st.info(answer)


# -----------------------------------------------------
# SHOW UPLOADED DOCUMENTS
# -----------------------------------------------------
st.subheader("Uploaded Documents")

if len(st.session_state.documents) == 0:
    st.write("No documents uploaded yet.")
else:
    for name, _ in st.session_state.documents:
        st.markdown(f"- **{name}**")
