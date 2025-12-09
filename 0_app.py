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

# ---------------- CSS + Dynamic Theme ----------------
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
        border-radius: 6px;
        border: 1px solid #eeeeee;
        white-space: pre-wrap;
    }}
    .bubble.ai p {{
        margin-top: 0.15rem;
        margin-bottom: 0.15rem;
    }}
    .bubble.ai ul {{
        margin-top: 0.15rem;
        margin-bottom: 0.15rem;
        padding-left: 1.2rem;
    }}
    .bubble.ai ul li {{
        margin-bottom: 0.1rem;
    }}

    .stButton > button {{
        border: 1px solid var(--theme-color);
        color: var(--theme-color);
    }}
    .stButton > button:hover {{
        background-color: var(--theme-color);
        color: white;
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

# Theme colors
theme_color = "#000000"
if tool == "Bayut":
    theme_color = "#008060"
elif tool == "Dubizzle":
    theme_color = "#D92C27"

st.markdown(get_theme_css(theme_color), unsafe_allow_html=True)

# Manage session keys
tool_key = tool.lower()
history_key = f"history_{tool_key}"
query_input_key = f"query_{tool_key}"

if history_key not in st.session_state:
    st.session_state[history_key] = []
if query_input_key not in st.session_state:
    st.session_state[query_input_key] = ""

# ---------------- FIXED PATH SYSTEM (WORKS IN CLOUD RUN) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder containing app.py
DATA_DIR = os.path.join(BASE_DIR, "data")               # /app/data in Cloud Run
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

st.write("📁 DATA DIR:", DATA_DIR)

# Debug: show files inside /data
if os.path.exists(DATA_DIR):
    st.write("📄 Files in data:", os.listdir(DATA_DIR))
else:
    st.write("❌ Data folder missing!")

# ---------------- File Matching ----------------
def find_best_matching_file(query: str):
    if not os.path.isdir(DATA_DIR):
        return None
    files = [f for f in os.listdir(DATA_DIR) if f != "faiss_store"]
    if not files:
        return None
    match = difflib.get_close_matches(
        query.lower(), [f.lower() for f in files], n=1, cutoff=0.3
    )
    if not match:
        return None
    for f in files:
        if f.lower() == match[0]:
            return os.path.join(DATA_DIR, f)
    return None

# ---------------- RAG Dependencies ----------------
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

# Local LLM (Groq)
def get_local_llm():
    if os.getenv("USE_DUMMY_LLM", "0") == "1":
        class DummyLLM:
            def invoke(self, text):
                return type("Resp", (), {"content": "This information is not available in internal content."})
        return DummyLLM()

    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0,
    )

# ---------------- Document Loading ----------------
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
            for enc in ["utf-8", "utf-8-sig", "cp1256", "windows-1256", None]:
                try:
                    return CSVLoader(path, encoding=enc).load()
                except:
                    continue
            return []
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [Document(page_content=df.to_string(index=False), metadata={"source": path})]
        return []
    except:
        return []

# ---------------- Build Vectorstore ----------------
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, get_embeddings())

# ---------------- Load / Rebuild FAISS Index ----------------
@st.cache_resource
def get_vectorstore():
    if os.path.exists(INDEX_DIR):
        try:
            return FAISS.load_local(
                INDEX_DIR,
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        except:
            pass

    store = build_vectorstore()
    if store:
        store.save_local(INDEX_DIR)
    return store

vectorstore = get_vectorstore()
if vectorstore is None:
    st.error("❌ No documents found in /data. Please add files and rebuild the index.")
    st.stop()

# ---------------- UI ----------------
st.write(f"### {tool}")

query = st.text_input("Ask your question:", key=query_input_key)

# Control buttons
col1, col2, col3 = st.columns(3)

def rebuild_index():
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    st.cache_resource.clear()
    st.experimental_rerun()

def clear_chat():
    st.session_state[history_key] = []
    st.session_state[query_input_key] = ""

def reload_app():
    st.experimental_rerun()

col1.button("Rebuild Index", on_click=rebuild_index)
col2.button("Clear Chat", on_click=clear_chat)
col3.button("Reload", on_click=reload_app)

# ---------------- Query Execution ----------------
if query.strip():
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    if isinstance(docs, dict) and "documents" in docs:
        docs = docs["documents"]

    context = "\n\n".join(d.page_content[:1800] for d in docs) if docs else ""

    history_items = st.session_state[history_key][-5:]
    history_text = "\n\n".join(
        f"User: {h['q']}\nAssistant: {h['a']}" for h in history_items
    ) or "No previous conversation."

    llm = get_local_llm()
    prompt = f"""
You are the Bayut & Dubizzle internal knowledge assistant.
Use ONLY the internal content in CONTEXT plus the CHAT HISTORY.

CHAT HISTORY:
{history_text}

CONTEXT:
{context}

NEW QUESTION:
{query}

Your task:
- If the answer exists in CONTEXT and/or CHAT HISTORY:
  produce JSON with "short_answer", "details", "source".
- Otherwise output:
  {{"short_answer":"This information is not available in internal content.","details":[],"source":""}}

Rules:
- JSON only.
- No markdown.
"""

    resp = llm.invoke(prompt)
    raw_text = resp.content if hasattr(resp, "content") else str(resp)

    try:
        data = json.loads(raw_text)
    except:
        data = {"short_answer": raw_text.strip(), "details": [], "source": ""}

    short_answer = data.get("short_answer", "").strip()
    details = data.get("details", [])
    if isinstance(details, str):
        details = [details]

    formatted = "Short Answer:\n" + short_answer
    if details:
        formatted += "\nDetails:\n" + "\n".join(f"- {d}" for d in details)

    st.session_state[history_key].append({"q": query, "a": formatted})

# ---------------- Display Chat ----------------
for item in reversed(st.session_state[history_key]):
    st.markdown(f"<div class='bubble user'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{item['a']}</div>", unsafe_allow_html=True)
