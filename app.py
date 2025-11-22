"""
Upgraded Read Smart AI app.py
Features:
- Multi-PDF upload
- Chunking & embeddings (HuggingFace)
- Chroma vector store
- RAG with Google Gemini via google.generativeai
- Chat history, conversation UI (dashboard + chat)
- PDF summary button
- Source citations + highlighting
- Export: Markdown (always), DOCX & PDF if libs installed (optional)
- Caching and lightweight design for Streamlit Cloud

REQUIREMENTS (add to requirements.txt):
streamlit
PyPDF2
pandas
google-generativeai
langchain-text-splitters
langchain-community
chromadb
sentence-transformers
python-docx (optional for DOCX export)
fpdf2 (optional for PDF export)

Make sure your secrets contain:
- GOOGLE_API_KEY
"""

import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
import io
from datetime import datetime

# LangChain-ish helpers (lightweight)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# For prompt templating
from textwrap import dedent

# Optional exports
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# -----------------------
# Helper utilities
# -----------------------
def init_genai():
    key = st.secrets.get("GOOGLE_API_KEY") if st.secrets else None
    if not key:
        st.error("Missing Google API key. Add GOOGLE_API_KEY to Streamlit secrets.")
        return False
    genai.configure(api_key=key)
    return True

def pdf_to_text(file):
    reader = PdfReader(file)
    texts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            texts.append({"page": i + 1, "text": t})
    return texts

def build_chunks_from_docs(docs_texts, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadata = []
    for doc in docs_texts:
        content = doc["text"]
        splits = splitter.split_text(content)
        for i, s in enumerate(splits):
            chunks.append(s)
            metadata.append({"source_name": doc.get("name", "uploaded_pdf"), "page": doc.get("page", None), "chunk_index": i})
    return chunks, metadata

def build_vectorstore(chunks, metadata, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Use HuggingFace embeddings via langchain community wrapper
    hf = HuggingFaceEmbeddings(model_name=embedding_model_name)
    # Use in-memory Chroma (persist_directory can be set to disk if you want persistence)
    chroma_ds = Chroma.from_texts(chunks, embedding=hf, metadatas=metadata)
    return chroma_ds

def retrieve_top_docs(vectorstore, query, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs

def call_gemini_with_context(prompt, model="gemini-1.5-flash", max_output_tokens=512):
    model_obj = genai.Models.get(model)  # safer getter
    # Use generate method
    resp = genai.generate(text=prompt, model=model)
    # google.generativeai returns complex objects; unify
    # resp will have .text or choices
    if hasattr(resp, "text"):
        return resp.text
    # try typical attr
    try:
        return resp["candidates"][0]["content"][0]["text"]
    except Exception:
        # fallback string conversion
        return str(resp)

# -----------------------
# Session state init
# -----------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "metadatas" not in st.session_state:
    st.session_state.metadatas = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {"role":"user/assistant","text":..., "sources":[...]}
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []  # list of uploaded doc descriptors

# -----------------------
# UI Layout (Dashboard + Chat)
# -----------------------
st.set_page_config(layout="wide", page_title="Read Smart AI - Dashboard")
st.title("Read Smart AI — Dashboard & Chat")

# Sidebar (filters & controls) - B: Professional dashboard controls
with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
    st.markdown("---")
    st.subheader("Embedding settings")
    chunk_size = st.number_input("Chunk size", value=1000, min_value=256, max_value=4000, step=128)
    chunk_overlap = st.number_input("Chunk overlap", value=200, min_value=0, max_value=1000, step=50)
    top_k = st.slider("Top k (retriever)", min_value=1, max_value=8, value=4)
    st.markdown("---")
    st.subheader("Actions")
    build_index_btn = st.button("Build / Rebuild Index")
    summarize_btn = st.button("Generate Summary for all docs")
    clear_btn = st.button("Clear all data")
    st.markdown("---")
    st.subheader("Export")
    export_md_btn = st.button("Export chat as Markdown")
    export_docx_btn = st.button("Export chat as DOCX" + ("" if HAS_DOCX else " (python-docx not installed)"))
    export_pdf_btn = st.button("Export chat as PDF" + ("" if HAS_FPDF else " (fpdf not installed)"))
    st.markdown("----")
    st.caption("Make sure GOOGLE_API_KEY is set in Streamlit secrets.")

# Main layout: left - chat area; right - docs & sources
col1, col2 = st.columns([3, 1])

with col1:
    # Chat-like message area (C: chat-app style)
    st.subheader("Conversation")
    # Display chat history as bubbles (simple)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div style='text-align:left; background:#eef3ff; padding:10px; border-radius:8px; margin:6px 0'>{msg['text']}</div>", unsafe_allow_html=True)
        else:
            # assistant
            sources_html = ""
            if msg.get("sources"):
                # build small citation badges
                badges = []
                for s in msg["sources"]:
                    src = f"{s.get('source_name','')}:p{s.get('page','?')}/c{s.get('chunk_index',0)}"
                    badges.append(f"<span style='background:#f0f0f0; padding:3px 6px; border-radius:6px; margin-right:4px; font-size:12px'>{src}</span>")
                sources_html = "<div style='margin-top:6px'>" + " ".join(badges) + "</div>"
            st.markdown(f"<div style='text-align:left; background:#f7f7f7; padding:10px; border-radius:8px; margin:6px 0'>{msg['text']}{sources_html}</div>", unsafe_allow_html=True)

    # Input area
    query = st.text_input("Ask a question about the uploaded PDFs:", key="query_input")
    ask_btn = st.button("Ask")

with col2:
    st.subheader("Uploaded Documents & Sources")
    if st.session_state.uploaded_docs:
        for d in st.session_state.uploaded_docs:
            st.markdown(f"**{d.get('name','unnamed')}** — pages: {d.get('pages', '?')}")
    else:
        st.info("No documents uploaded yet. Use the sidebar to upload PDFs.")

    # Show a small preview of top sources if last answer exists
    if st.session_state.chat_history and st.session_state.chat_history[-1].get("sources"):
        st.markdown("---")
        st.markdown("**Top source snippets (from last answer)**")
        for i, s in enumerate(st.session_state.chat_history[-1]["sources"]):
            snippet = s.get("text","")[:300].replace("\n"," ")
            st.markdown(f"- **{s.get('source_name','')}, page {s.get('page','?')}**: {snippet}...")

# -----------------------
# Handlers
# -----------------------
# Upload handling: when files are selected, parse them and store in session_state.uploaded_docs
if uploaded_files:
    new_docs = []
    for f in uploaded_files:
        try:
            texts = pdf_to_text(f)
            # store each page item as separate doc with metadata name+page
            for page_item in texts:
                new_docs.append({"name": getattr(f, "name", "uploaded_pdf"), "page": page_item["page"], "text": page_item["text"]})
        except Exception as e:
            st.error(f"Failed to read {getattr(f,'name','file')}: {e}")
    if new_docs:
        st.session_state.uploaded_docs = new_docs
        st.success(f"Loaded {len(new_docs)} pages from uploaded files.")

# Clear all
if clear_btn:
    st.session_state.vector_store = None
    st.session_state.chunks = []
    st.session_state.metadatas = []
    st.session_state.chat_history = []
    st.session_state.uploaded_docs = []
    st.success("Cleared all data and history.")

# Build / rebuild index
if build_index_btn:
    if not st.session_state.uploaded_docs:
        st.warning("No documents to index. Upload PDFs first.")
    else:
        with st.spinner("Building index..."):
            chunks, metadatas = build_chunks_from_docs(st.session_state.uploaded_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.session_state.chunks = chunks
            st.session_state.metadatas = metadatas
            st.session_state.vector_store = build_vectorstore(chunks, metadatas)
        st.success(f"Indexed {len(chunks)} chunks.")

# Summarize all docs
if summarize_btn:
    if not init_genai():
        st.stop()
    if not st.session_state.uploaded_docs:
        st.warning("Upload PDFs before summarizing.")
    else:
        with st.spinner("Generating summary..."):
            # build a combined text sample for summary (first N characters to stay small)
            combined = "\n\n".join([d["text"] for d in st.session_state.uploaded_docs])
            prompt = dedent(f"""
            You are an assistant. Provide a concise summary (3-6 bullet points) of the text below.

            Text:
            {combined[:20000]}

            Bulleted summary:
            """)
            out = call_gemini_with_context(prompt)
        st.session_state.chat_history.append({"role":"assistant","text":out, "sources":[]})
        st.success("Summary generated and added to chat history.")

# Export handlers
def export_chat_as_markdown(history):
    md = []
    md.append(f"# Read Smart AI Conversation Export")
    md.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")
    for m in history:
        role = m["role"]
        md.append(f"## {role.upper()}\n")
        md.append(m["text"] + "\n")
        if m.get("sources"):
            md.append("### SOURCES\n")
            for s in m["sources"]:
                md.append(f"- {s.get('source_name')} (page {s.get('page')})\n")
    return "\n".join(md).encode("utf-8")

def export_chat_as_docx(history):
    if not HAS_DOCX:
        raise RuntimeError("python-docx not installed")
    doc = DocxDocument()
    doc.add_heading("Read Smart AI Conversation Export", level=1)
    doc.add_paragraph(f"Generated: {datetime.utcnow().isoformat()}Z")
    for m in history:
        doc.add_heading(m["role"].upper(), level=2)
        doc.add_paragraph(m["text"])
        if m.get("sources"):
            doc.add_heading("Sources", level=3)
            for s in m["sources"]:
                doc.add_paragraph(f"- {s.get('source_name')} (page {s.get('page')})")
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

def export_chat_as_pdf(history):
    if not HAS_FPDF:
        raise RuntimeError("fpdf not installed")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, txt="Read Smart AI Conversation Export", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    for m in history:
        pdf.multi_cell(0, 6, txt=f"{m['role'].upper()}:\n{m['text']}")
        pdf.ln(2)
        if m.get("sources"):
            pdf.multi_cell(0, 6, txt="Sources:")
            for s in m["sources"]:
                pdf.multi_cell(0, 6, txt=f"- {s.get('source_name')} (page {s.get('page')})")
            pdf.ln(2)
    bio = io.BytesIO()
    pdf.output(bio)
    bio.seek(0)
    return bio.read()

# Export buttons (these trigger downloads)
if export_md_btn:
    data = export_chat_as_markdown(st.session_state.chat_history)
    st.download_button("Download conversation (MD)", data=data, file_name="read_smart_ai_conversation.md", mime="text/markdown")
if export_docx_btn:
    try:
        data = export_chat_as_docx(st.session_state.chat_history)
        st.download_button("Download conversation (DOCX)", data=data, file_name="read_smart_ai_conversation.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    except Exception as e:
        st.error(f"DOCX export failed: {e}")
if export_pdf_btn:
    try:
        data = export_chat_as_pdf(st.session_state.chat_history)
        st.download_button("Download conversation (PDF)", data=data, file_name="read_smart_ai_conversation.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF export failed: {e}")

# -----------------------
# Main ask handler
# -----------------------
if ask_btn or (query and st.session_state.get("auto_ask_on_enter", False)):
    if not query or not query.strip():
        st.warning("Please enter a question.")
    else:
        if not st.session_state.vector_store:
            # try to auto-build if uploaded_docs present
            if st.session_state.uploaded_docs:
                with st.spinner("Auto-building index..."):
                    chunks, metadatas = build_chunks_from_docs(st.session_state.uploaded_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.session_state.chunks = chunks
                    st.session_state.metadatas = metadatas
                    st.session_state.vector_store = build_vectorstore(chunks, metadatas)
                st.success("Index built.")
            else:
                st.warning("No index available. Upload PDFs and build the index.")
                st.stop()

        # Retrieve top docs
        docs = retrieve_top_docs(st.session_state.vector_store, query, k=top_k)
        # prepare context (include small snippets and metadata)
        taken = []
        for d in docs:
            meta = d.metadata if hasattr(d, "metadata") else {}
            taken.append({"source_name": meta.get("source_name","uploaded_pdf"), "page": meta.get("page"), "chunk_index": meta.get("chunk_index"), "text": d.page_content})

        context = "\n\n---\n\n".join([f"Source: {t['source_name']} (page {t['page']})\n\n{t['text']}" for t in taken])

        # Build prompt with instructions to only use context
        prompt = dedent(f"""
        You are an expert assistant. Use ONLY the context below to answer the user's question. If the answer isn't in the context, say you don't know.

        CONTEXT:
        {context}

        USER QUESTION:
        {query}

        ANSWER (concise, reference the sources used by adding [source_name:page] after facts):
        """)

        if not init_genai():
            st.stop()

        with st.spinner("Generating answer from Gemini..."):
            # Call Google Gemini with prompt
            try:
                resp = genai.generate(model="gemini-1.5-flash", prompt=prompt)
                # The response API might differ; unify access:
                answer_text = ""
                if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                    # google response object shape
                    answer_text = resp.candidates[0].content[0].text if hasattr(resp.candidates[0].content[0], "text") else str(resp)
                else:
                    # fallback
                    answer_text = str(resp)
            except Exception as e:
                # Try alternate simple generate_content call
                try:
                    model_obj = genai.Models.get("gemini-1.5-flash")
                    g = genai.generate(model="gemini-1.5-flash", prompt=prompt)
                    answer_text = str(g)
                except Exception as ee:
                    st.error(f"LLM call failed: {ee}")
                    st.stop()

        # Save user message and assistant response in history with sources
        st.session_state.chat_history.append({"role":"user","text":query, "sources":[]})
        st.session_state.chat_history.append({"role":"assistant","text":answer_text, "sources":taken})

        # Scroll to bottom by rerendering (Streamlit auto)
        st.experimental_rerun()
