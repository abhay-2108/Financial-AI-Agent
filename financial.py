import sys
import os

# Fix for chromadb sqlite3 version issue
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    raise ImportError("pysqlite3 is required but not installed. Add 'pysqlite3-binary' to requirements.txt.")

import streamlit as st
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredFileLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough


st.set_page_config(page_title="FinAgent - Financial Chatbot", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None

MODEL_NAME = "llama3.1:8b"
llm = OllamaLLM(model=MODEL_NAME, temperature=0.1)
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>FinAgent - Financial Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload financial documents (PDF, Excel, CSV, etc.) and chat with them using AI</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(" ", type=["pdf", "csv", "txt", "html", "htm", "xlsx"], label_visibility="collapsed")

def detect_and_load(file):
    ext = os.path.splitext(file.name)[1].lower()
    with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.read())
        temp_path = tmp.name

    if ext == ".pdf":
        loader = PyMuPDFLoader(temp_path)
    elif ext == ".csv":
        loader = CSVLoader(temp_path)
    elif ext in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(temp_path)
    elif ext in [".txt", ".html", ".htm"]:
        loader = UnstructuredFileLoader(temp_path)
    else:
        st.error(f"Unsupported file type: {ext}")
        return []
    return loader.load()

if uploaded_file:
    st.markdown(f"<small style='color: gray;'>📎 File attached: <code>{uploaded_file.name}</code></small>", unsafe_allow_html=True)
    docs = detect_and_load(uploaded_file)

    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(
            """
You are **FinAgent**, a highly intelligent and reliable financial assistant AI trained to analyze and answer questions based on financial documents.

Context from the document:
====================
{context}
====================

Question: {question}

Answer:
"""
        )

        rag_chain = RunnableMap({
            "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
            "question": RunnablePassthrough(),
        }) | prompt | llm

        st.session_state.rag_chain = rag_chain
        st.session_state.chat_history = []  


with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input("Type your question...", label_visibility="collapsed", placeholder="e.g., What was the EBITDA last year?")
    with col2:
        submitted = st.form_submit_button("Send", use_container_width=True)


if submitted and user_input:
    if not uploaded_file:
        st.warning("Please upload a file first.")
    elif not st.session_state.rag_chain:
        st.warning("Something went wrong loading the document.")
    else:
        with st.spinner("Analyzing..."):
            st.session_state.chat_history.append(("You", user_input))
            response = st.session_state.rag_chain.invoke(user_input)
            final_answer = response if isinstance(response, str) else str(response)
            st.session_state.chat_history.append(("FinAgent", final_answer))


with st.container():
    st.markdown("""
        <style>
        .chat-box {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #f8f9fa;
            margin-bottom: 15px;
        }
        .user-msg {
            text-align: right;
            background-color: #d1e7dd;
            color: #000;
            display: block;
            padding: 10px 15px;
            margin: 5px 0 5px auto;
            border-radius: 10px;
            max-width: 80%;
            margin-left: auto;
        }
        .ai-msg {
            text-align: left;
            background-color: #e2e3e5;
            color: #000;
            display: block;
            padding: 10px 15px;
            margin: 5px auto 5px 0;
            border-radius: 10px;
            max-width: 80%;
            margin-right: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for speaker, message in st.session_state.chat_history:
        css_class = "user-msg" if speaker == "You" else "ai-msg"
        st.markdown(f"<div class='{css_class}'><strong>{speaker}:</strong><br>{message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
