import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

# Streamlit UI
st.set_page_config(page_title="Read Smart AI", layout="wide")
st.title("📘 Read Smart AI – PDF Question Answering")

# Google API key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("PDF loaded successfully!")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    # Embeddings (HuggingFace works on Streamlit Cloud)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chroma vector store
    vector_store = Chroma.from_texts(chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # User input
    query = st.text_input("Ask something about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

        st.subheader("Answer:")
        st.write(response.text)
