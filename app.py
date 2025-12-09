# =========================================================
# Bayut & Dubizzle AI Content Assistant — Internal RAG Only
# =========================================================

import os
import shutil
import difflib
import json
import pandas as pd
import streamlit as st
from uuid import uuid4
from langchain_core.documents import Document

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide",
)

# ---------------- CSS ----------------
def get_theme_css(color: str) -> str:
    return f"""
    <style>
    :root {{
        --theme-color: {color};
    }}
    h2, h3, h4, label {{
        color: var(--theme-color) !important;
    }}
    .bubble {{
        padding: 10px 14px;
        border-radius: 6px;
        margin: 4px 0;
        max-width: 100%;
        line-height: 1.6;
        font-size: 15px;
    }}
    .bubble.user {{
        background: #f5f5f5;
        border-left: 3px solid var(--theme-color);
    }}
    .bubble.ai {{
        background: #ffffff;
        border: 1px solid #eee;
        white-space: pre-wrap;
    }}
    </style>
    """

# ---------------- Header ----------------
st.markdown(
    """
<div style='text-align:center; font-size:38px; font-weight:900; margin-bottom:0;'>
  <span style='color:#008060;'>Bayut</span>
  <span style='color:#000000;'> & </span>
  <span style='color:#D92C27;'>Dubizzle</span>
  <span style='color:#000000;'> AI Content Assistant</span>
</div>
<p style='text-align:center; color:#555; margin-top:-6px; font-size:15px;'>
Fast internal knowledge search powered by internal content.
</p>
""",
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
st.sidebar.markdown("#### Select an option")
tool = st.sidebar.radio("", ["General", "Bayut", "Dubizzle"])

theme_color = "#000000"
if tool == "Bayut":
    theme_color = "#008060"
elif tool == "Dubizzle":
    theme_color = "#D92C27"

st.markdown(get_theme_css(theme_color), unsafe_allow_html=True)

# session states
tool_key = tool.lower()
history_key = f"history_{tool_key}"
query_key = f"query_{tool_key}"

if history_key not in st.session_state:
    st.session_state[history_key] = []

if query_key not in st.session_state:
    st.session_state[query_key] = ""

# ---------------- Data paths ----------------
DATA_DIR = "/app/data" if os.getenv("CLOUD_RUN") else "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

st.caption(f"📁 DATA DIR: **{DATA_DIR}**")

# show files
if os.path.isdir(DATA_DIR):
    st.json(os.listdir(DATA_DIR))

# ---------------- Loaders ----------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_local_llm():
    """Groq LLM wrapper"""
    if os.getenv("USE_DUMMY_LLM", "0") == "1":
        class DummyLLM:
            def invoke(self, text):
                return type("Resp", (), {"content": "Dummy model active. No real response."})
        return DummyLLM()

    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0,
    )

def load_document(path: str):
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        if ext == ".docx":
            return Docx2txtLoader(path).load()
        if ext in [".txt", ".md"]:
            return TextLoader(path, autodetect_encoding=True).load()
        if ext == ".csv":
            return CSVLoader(path, encoding="utf-8").load()
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [Document(page_content=df.to_string(index=False), metadata={"source": path})]
        return []
    except:
        return []

def build_vectorstore():
    docs = []
    if not os.path.isdir(DATA_DIR):
        return None

    for f in os.listdir(DATA_DIR):
        if f == "faiss_store":
            continue
        full = os.path.join(DATA_DIR, f)
        if os.path.isfile(full):
            docs.extend(load_document(full))

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, get_embeddings())

# ---------------- Load / Rebuild Index ----------------
@st.cache_resource
def get_vectorstore():
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    store = build_vectorstore()
    if store:
        store.save_local(INDEX_DIR)
    return store

vectorstore = get_vectorstore()

if vectorstore is None:
    st.error("❌ No documents found in /data. Please upload files and click **Rebuild Index**.")
    st.stop()

# -------------------------------------------
# Query Input
# -------------------------------------------
st.write(f"### {tool}")

query = st.text_input("Ask your question:", key=query_key)

col1, col2, col3 = st.columns(3)

def rebuild_index():
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    st.cache_resource.clear()
    st.experimental_rerun()

def clear_chat():
    st.session_state[history_key] = []
    st.session_state[query_key] = ""

with col1:
    st.button("Rebuild Index", on_click=rebuild_index)
with col2:
    st.button("Clear Chat", on_click=clear_chat)
with col3:
    st.button("Reload", on_click=st.experimental_rerun)

# -------------------------------------------
# Run Query
# -------------------------------------------
if query.strip():
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(d.page_content[:1500] for d in docs)

    history_text = "\n".join(
        f"User: {h['q']}\nAssistant: {h['a']}"
        for h in st.session_state[history_key][-5:]
    ) or "No previous history."

    llm = get_local_llm()

    prompt = f"""
You are the Bayut & Dubizzle Internal Knowledge Assistant.
Use ONLY the text inside CONTEXT and CHAT HISTORY.

CHAT HISTORY:
{history_text}

CONTEXT:
{context}

NEW QUESTION:
{query}

Return JSON:
- short_answer: one clear sentence
- details: bullet points
- source: file name
"""

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        data = json.loads(raw)
    except:
        data = {"short_answer": raw, "details": [], "source": ""}

    formatted = f"Short Answer:\n{data['short_answer']}\n\n"
    if data.get("details"):
        formatted += "Details:\n" + "\n".join(f"- {d}" for d in data["details"])

    st.session_state[history_key].append({"q": query, "a": formatted})

# -------------------------------------------
# Display chat
# -------------------------------------------
for item in reversed(st.session_state[history_key]):
    st.markdown(f"<div class='bubble user'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{item['a']}</div>", unsafe_allow_html=True)
