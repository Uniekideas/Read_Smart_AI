import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

# Correct LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


# ---------------------- PDF PROCESSING ----------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


# ---------------------- CHAIN SETUP ----------------------
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say: "answer is not available in the context".
    Provide bullet points and paragraphs when needed.

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

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    return chain


# ---------------------- QUERY PROCESSING ----------------------
def user_input(user_question, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files and provide an API key before processing.")
        return

    # Create chunks
    text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text)

    # Vector store
    vector_store = get_vector_store(text_chunks, api_key)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve docs
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    user_question_output = user_question
    response_output = response['output_text']

    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    conversation_history.append(
        (user_question_output, response_output, "Google AI",
         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
         ", ".join(pdf_names))
    )

    # Display messages
    display_messages(user_question_output, response_output, conversation_history)


def display_messages(user_q, bot_resp, conversation_history):
    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.2rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{ background-color: #2b313e; }}
            .chat-message.bot {{ background-color: #475063; }}
            .chat-message .avatar img {{
                max-width: 70px;
                max-height: 70px;
                border-radius: 50%;
            }}
            .chat-message .message {{
                width: 80%;
                padding-left: 1rem;
                color: #fff;
            }}
        </style>

        <div class="chat-message user">
            <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
            <div class="message">{user_q}</div>
        </div>

        <div class="chat-message bot">
            <div class="avatar"><img src="https://i.ibb.co/pvv9dmdS/langchain-logo.webp"></div>
            <div class="message">{bot_resp}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # List previous messages
    for question, answer, model, timestamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
                <div class="message">{question}</div>
            </div>

            <div class="chat-message bot">
                <div class="avatar"><img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"></div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Export CSV
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    st.snow()


# ---------------------- STREAMLIT UI ----------------------
def main():
    st.set_page_config(page_title="Read_Smart_AI", page_icon="📚")
    st.header("Read_Smart_AI (v1) 📚")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    model_name = "Google AI"  # only model enabled

    api_key = st.sidebar.text_input("Enter your Google API Key:")
    st.sidebar.markdown("Get an API Key at https://ai.google.dev/")

    if not api_key:
        st.sidebar.warning("Please enter your API key to continue.")
        return

    pdf_docs = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                st.success("PDFs processed successfully!")
        else:
            st.sidebar.warning("Upload at least one PDF.")

    user_question = st.text_input("Ask me anything about your PDF:")

    if user_question:
        user_input(user_question, api_key, pdf_docs, st.session_state.conversation_history)

    # Reset Button
    if st.sidebar.button("Reset Everything"):
        st.session_state.conversation_history = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
