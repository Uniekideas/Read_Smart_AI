import streamlit as st
from PyPDF2 import PdfReader

# Updated LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.chains import RetrievalQA

# Streamlit UI
st.set_page_config(page_title="Read Smart AI", layout="wide")
st.title("📘 Read Smart AI – PDF Question Answering")

uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("PDF loaded successfully! 🎉")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Vector store (Chroma instead of FAISS)
    vector_store = Chroma.from_texts(chunks, embedding=embeddings)

    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Chat model
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Retrieval Q&A Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # User question input
    query = st.text_input("Ask something about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)

        st.subheader("Answer:")
        st.write(answer)
