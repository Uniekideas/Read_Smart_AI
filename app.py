import streamlit as st
from PyPDF2 import PdfReader
from datetime import datetime
import pandas as pd
import base64

# LangChain imports (correct for v0.2+)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# NEW retrieval chain imports
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -----------------------------
# PDF Extractor
# -----------------------------
def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# -----------------------------
# Chunk Text
# -----------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=800
    )
    return splitter.split_text(text)


# -----------------------------
# Vector Store
# -----------------------------
def build_vectorstore(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(chunks, embedding=embeddings)


# -----------------------------
# Build Retrieval Chain
# -----------------------------
def build_qa_chain(api_key, vectorstore):

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You must answer ONLY using the provided context.

If the answer is not found in the context, respond:
"Answer is not available in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

    # Stuff chain (for merging docs)
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Retrieval chain (search + LLM)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=document_chain
    )

    return retrieval_chain


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Read Smart AI", page_icon="📘")
    st.title("📘 Read Smart AI — PDF Analyzer")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    api_key = st.sidebar.text_input("Google API Key", type="password")

    if not api_key:
        st.stop()

    pdf_docs = st.sidebar.file_uploader(
        "Upload PDF(s)",
        accept_multiple_files=True
    )

    if st.sidebar.button("Process PDFs"):
        if not pdf_docs:
            st.sidebar.warning("Upload a PDF")
        else:
            with st.spinner("Processing..."):
                text = extract_pdf_text(pdf_docs)
                chunks = chunk_text(text)
                st.session_state.vectorstore = build_vectorstore(chunks, api_key)
            st.sidebar.success("PDFs indexed successfully!")

    st.write("---")

    # Question input
    question = st.text_input("Ask a question about your documents:")

    if question:
        if "vectorstore" not in st.session_state:
            st.warning("Upload and process PDFs first.")
            st.stop()

        qa_chain = build_qa_chain(api_key, st.session_state.vectorstore)
        response = qa_chain.invoke({"question": question})

        answer = response["answer"]

        st.session_state.history.append(
            (question, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        st.markdown(
            f"<div style='padding:1rem;background:#2c2f3b;border-radius:8px;color:white;'><b>You:</b> {question}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='padding:1rem;background:#3d4250;border-radius:8px;color:white;'><b>AI:</b> {answer}</div>",
            unsafe_allow_html=True
        )

    # History download
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history, columns=["Question", "Answer", "Timestamp"])
        csv = df.to_csv(index=False)
        st.download_button("Download session history", csv, "history.csv")


if __name__ == "__main__":
    main()
