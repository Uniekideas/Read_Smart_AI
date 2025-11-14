import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

# ---------------------------
# LangChain Updated Imports
# ---------------------------

# TEXT SPLITTER
from langchain_text_splitters import RecursiveCharacterTextSplitter

# GOOGLE GEN AI (Embeddings + Chat Model)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# VECTORSTORE
from langchain_community.vectorstores import FAISS

# PROMPTS
from langchain.prompts import PromptTemplate

# NEW RAG CHAIN SYSTEM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -----------------------------------------
# PDF TEXT EXTRACTION
# -----------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# -----------------------------------------
# TEXT SPLITTING
# -----------------------------------------
def get_text_chunks(text, model_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


# -----------------------------------------
# VECTOR STORE CREATION
# -----------------------------------------
def get_vector_store(text_chunks, model_name, api_key=None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


# -----------------------------------------
# BUILD THE NEW LANGCHAIN RAG PIPELINE
# -----------------------------------------
def get_conversational_chain(model_name, vectorstore=None, api_key=None):

    prompt_template = """
    You are a helpful AI assistant.

    Answer the question strictly using the information provided in the context.
    - If the answer is NOT present in the context, say: "answer is not available in the context."
    - Provide answers in clear paragraphs and bullet points when needed.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        google_api_key=api_key
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Build document chain
    document_chain = create_stuff_documents_chain(model, prompt)

    # Build retriever
    retriever = vectorstore.as_retriever()

    # COMPLETE RAG PIPELINE
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain


# -----------------------------------------
# PROCESS USER INPUT
# -----------------------------------------
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):

    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files and enter your API key.")
        return

    # Process PDFs → chunks → vector DB
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chunks, model_name, api_key)

    # Load FAISS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    chain = get_conversational_chain(model_name, new_db, api_key)

    # RUN THE CHAIN
    response = chain.invoke({"question": user_question})
    answer = response["answer"]

    # Save history
    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    conversation_history.append(
        (
            user_question,
            answer,
            model_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ", ".join(pdf_names)
        )
    )

    # ---------------------------
    # CHAT UI DISPLAY
    # ---------------------------

    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar img {{
                max-width: 70px;
                border-radius: 50%;
            }}
            .chat-message .message {{
                padding-left: 1rem;
                color: white;
            }}
        </style>

        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>
            <div class="message">{user_question}</div>
        </div>

        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/pvv9dmdS/langchain-logo.webp">
            </div>
            <div class="message">{answer}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Render past conversation
    for q, a, model, timestamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">{q}</div>
            </div>

            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{a}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # DOWNLOAD CHAT HISTORY
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.sidebar.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download Conversation</button></a>',
            unsafe_allow_html=True
        )


# -----------------------------------------
# MAIN UI
# -----------------------------------------
def main():
    st.set_page_config(page_title="Read_Smart_AI", page_icon="📚")
    st.header("Read_Smart_AI (v1) 📚")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar
    st.sidebar.title("Menu")

    model_name = "Google AI"  # only model you support now

    api_key = st.sidebar.text_input("Enter your Google API Key:")
    st.sidebar.markdown("[Get API Key](https://ai.google.dev/)")

    if not api_key:
        st.sidebar.warning("Google API key is required.")
        return

    col1, col2 = st.sidebar.columns(2)

    if col1.button("Rerun"):
        if st.session_state.conversation_history:
            st.session_state.conversation_history.pop()

    if col2.button("Reset"):
        st.session_state.conversation_history = []

    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF files",
        accept_multiple_files=True
    )

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                st.success("PDFs processed.")
        else:
            st.warning("Upload PDFs first.")

    # USER QUESTION
    user_question = st.text_input("Ask me anything about your PDFs")

    if user_question:
        user_input(
            user_question,
            model_name,
            api_key,
            pdf_docs,
            st.session_state.conversation_history
        )


if __name__ == "__main__":
    main()
