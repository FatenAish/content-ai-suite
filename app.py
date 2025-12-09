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

theme_color = "#000000"
if tool == "Bayut":
    theme_color = "#008060"
elif tool == "Dubizzle":
    theme_color = "#D92C27"

st.markdown(get_theme_css(theme_color), unsafe_allow_html=True)

# Per-tool keys
tool_key = tool.lower()
history_key = f"history_{tool_key}"
query_input_key = f"query_{tool_key}"

if history_key not in st.session_state:
    st.session_state[history_key] = []
if query_input_key not in st.session_state:
    st.session_state[query_input_key] = ""

# ---------------- FIXED FILE PATHS (WORK ON STREAMLIT CLOUD + LOCAL + DOCKER) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # Folder where app.py lives
DATA_DIR = os.path.join(BASE_DIR, "data")                   # Correct data path
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")           # Where we store FAISS index

st.write("📁 DATA DIR:", DATA_DIR)

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

# ---------------- LangChain RAG ----------------
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
                except Exception:
                    continue
            return []
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [Document(page_content=df.to_string(index=False), metadata={"source": path})]
        return []
    except Exception:
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())

# ---------------- Load or Build FAISS Index ----------------
@st.cache_resource
def get_vectorstore():
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    store = build_vectorstore()
    if store is not None:
        store.save_local(INDEX_DIR)
    return store

vectorstore = get_vectorstore()
if vectorstore is None:
    st.error("❌ No documents found in the data folder. Please add files in /data and rebuild the index.")
    st.stop()

# ---------------- Title ----------------
st.write(f"### {tool}")

# ---------------- Query ----------------
query = st.text_input("Ask your question:", key=query_input_key)

col_spacer_l, col1, col2, col3, col_spacer_r = st.columns([2, 1, 1, 1, 2])

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

with col1: st.button("Rebuild Index", on_click=rebuild_index)
with col2: st.button("Clear Chat", on_click=clear_chat)
with col3: st.button("Reload", on_click=reload_app)

# ---------------- Run Query ----------------
if query.strip():
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    if isinstance(docs, dict) and "documents" in docs:
        docs = docs["documents"]

    context = "\n\n".join(d.page_content[:1800] for d in docs) if docs else ""

    history_items = st.session_state[history_key][-5:]
    history_snippets = [
        f"User: {h['q']}\nAssistant: {h['a']}" for h in history_items
    ]
    history_text = "\n\n".join(history_snippets) if history_snippets else "No previous conversation."

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
- One clear sentence short answer.
"""

    resp = llm.invoke(prompt)
    raw_text = resp.content if hasattr(resp, "content") else str(resp)

    try:
        data = json.loads(raw_text)
    except Exception:
        data = {"short_answer": raw_text.strip(), "details": [], "source": ""}

    short_answer = str(data.get("short_answer", "")).strip()
    details = data.get("details", [])
    if isinstance(details, str):
        details = [details] if details.strip() else []
    details = [str(d).strip() for d in details if str(d).strip()]

    lines = []
    if short_answer:
        lines.append("Short Answer:")
        lines.append(short_answer)
    if details:
        lines.append("Details:")
        for d in details:
            lines.append(f"- {d}")

    formatted = "\n".join(lines).strip()
    if not formatted:
        formatted = "Short Answer:\nThis information is not available in internal content."

    st.session_state[history_key].append({"q": query, "a": formatted})

# ---------------- Show Chat ----------------
for item in reversed(st.session_state[history_key]):
    st.markdown(f"<div class='bubble user'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{item['a']}</div>", unsafe_allow_html=True)
