# =========================================================
# Bayut & Dubizzle AI Content Assistant — Internal RAG Only
# =========================================================

import os
import shutil
import difflib
import json
import pandas as pd
import streamlit as st
from langchain_core.documents import Document

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
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

    .bubble.ai {{
        padding: 8px 12px;
        margin-top: 10px;
        background: #ffffff;
        border-radius: 6px;
        border: 1px solid #eee;
        white-space: pre-wrap;
    }}

    .stButton>button {{
        border: 1px solid var(--theme-color);
        color: var(--theme-color);
    }}

    .stButton>button:hover {{
        background-color: var(--theme-color);
        color: white;
    }}
    </style>
    """

# ---------------- Header ----------------
st.markdown("""
<div style='text-align:center; font-size:38px; font-weight:900; margin-bottom:0;'>
    <span style='color:#008060;'>Bayut</span>
    <span style='color:#000000;'> & </span>
    <span style='color:#D92C27;'>Dubizzle</span>
    <span style='color:#000000;'> AI Content Assistant</span>
</div>

<p style='text-align:center; color:#555; margin-top:-6px; font-size:15px;'>
Fast internal knowledge search powered by internal content.
</p>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.markdown("#### Select an option")
tool = st.sidebar.radio("", ["General", "Bayut", "Dubizzle"])

theme_color = "#000000"
if tool == "Bayut":
    theme_color = "#008060"
elif tool == "Dubizzle":
    theme_color = "#D92C27"

st.markdown(get_theme_css(theme_color), unsafe_allow_html=True)

# ---------------- Paths ----------------
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

# ---------------- Helper: fuzzy file search (kept for future use) ----------------
def find_best_matching_file(query: str):
    if not os.path.isdir(DATA_DIR):
        return None
    files = [f for f in os.listdir(DATA_DIR) if f != "faiss_store"]
    if not files:
        return None
    match = difflib.get_close_matches(query.lower(), [f.lower() for f in files], n=1, cutoff=0.3)
    if not match:
        return None
    for f in files:
        if f.lower() == match[0]:
            return os.path.join(DATA_DIR, f)
    return None

# ---------------- LangChain / RAG ----------------
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
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0
    )

def load_document(path: str):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        if ext == ".docx":
            return Docx2txtLoader(path).load()
        if ext in [".txt", ".md"]:
            return TextLoader(path).load()
        if ext == ".csv":
            return CSVLoader(path).load()
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [Document(page_content=df.to_string(index=False))]
        return []
    except Exception:
        return []

def build_vectorstore():
    if not os.path.isdir(DATA_DIR):
        return None
    docs = []
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

# ---------------- Load FAISS index ----------------
if os.path.exists(INDEX_DIR):
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
else:
    vectorstore = build_vectorstore()
    if vectorstore:
        vectorstore.save_local(INDEX_DIR)

# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state["history"] = []      # list of {"q":..., "a":...}
if "last_processed_query" not in st.session_state:
    st.session_state["last_processed_query"] = ""

# ---------------- UI: tool title ----------------
st.write(f"### {tool}")

# ---------------- Question box ----------------
query = st.text_input("Ask your question:")

# Detect a new question (so we don't answer the same text twice)
is_new_question = (
    query
    and query.strip()
    and query != st.session_state["last_processed_query"]
)

# ---------------- Run RAG only when we have a NEW query ----------------
if is_new_question and vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    # newer LangChain sometimes returns dict with "documents"
    if isinstance(docs, dict) and "documents" in docs:
        docs = docs["documents"]

    context = "\n\n".join(d.page_content[:1800] for d in docs) if docs else ""

    llm = get_local_llm()

    # Build short chat history for LLM (last 5 turns)
    history_snippets = []
    for h in st.session_state["history"][-5:]:
        history_snippets.append(f"User: {h['q']}\nAssistant: {h['a']}")
    history_text = "\n\n".join(history_snippets) if history_snippets else "No previous conversation."

    # Model returns JSON: short_answer + optional details[]
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
- If the answer exists in CONTEXT and/or is implied by the previous answers in CHAT HISTORY,
  produce a JSON object with:
  - "short_answer": one direct sentence answering the question.
  - "details": a list of 0–3 short explanation strings (optional).
  - "source": short reference to document name or date, if visible in context. Empty string if unknown.

- If the answer does NOT exist in CONTEXT or CHAT HISTORY, return exactly this JSON:
  {{"short_answer": "This information is not available in internal content.", "details": [], "source": ""}}

Style rules:
- DO NOT mention "chat history", previous questions, Q1/Q2 etc.
- Just answer naturally using the information.
- The short answer must be one clear sentence.
- Details must be short, practical support sentences.

Output rules:
- Output JSON ONLY. No markdown, no labels, no explanation.
- JSON must be valid and parseable by Python json.loads().
JSON ANSWER:
"""

    resp = llm.invoke(prompt)

    # ChatGroq returns an AIMessage; get .content safely
    if hasattr(resp, "content"):
        raw_text = resp.content
    else:
        raw_text = str(resp)

    # Try to parse JSON
    try:
        data = json.loads(raw_text)
    except Exception:
        data = {
            "short_answer": raw_text.strip(),
            "details": [],
            "source": ""
        }

    short_answer = str(data.get("short_answer", "")).strip()
    details = data.get("details", [])
    if isinstance(details, str):
        details = [details] if details.strip() else []
    details = [str(d).strip() for d in details if str(d).strip()]

    # ---------- Build answer: Short Answer then Details ----------
    lines = []
    if short_answer:
        lines.append("Short Answer:")
        lines.append(short_answer)
    if details:
        lines.append("")
        lines.append("Details:")
        for d in details:
            lines.append(d)

    formatted = "\n".join(lines).strip()
    if not formatted:
        formatted = "Short Answer:\nThis information is not available in internal content."

    # Save to history
    st.session_state["history"].append({"q": query, "a": formatted})
    st.session_state["last_processed_query"] = query

# ---------------- Show ONLY the answers (no question bubbles) ----------------
for item in st.session_state["history"]:
    st.markdown(
        f"<div class='bubble ai'>{item['a']}</div>",
        unsafe_allow_html=True,
    )

# ---------------- Buttons row (bottom) ----------------
st.write("")  # small spacer
left, col1, col2, col3, right = st.columns([2, 1, 1, 1, 2])

with col1:
    b1 = st.button("Rebuild Index")
with col2:
    b2 = st.button("Clear Chat")
with col3:
    b3 = st.button("Reload")

if b1:
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    st.cache_resource.clear()
    st.session_state["history"] = []
    st.session_state["last_processed_query"] = ""
    st.success("Index cleared. Refresh the page.")

if b2:
    st.session_state["history"] = []
    st.session_state["last_processed_query"] = ""

if b3:
    st.rerun()
