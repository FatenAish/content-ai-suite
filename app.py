# =========================================================
# Bayut & Dubizzle AI Content Assistant — Internal RAG Only
# =========================================================

import os
import shutil
import difflib
import pandas as pd
import streamlit as st
from uuid import uuid4
from langchain_core.documents import Document


# ---------------- Page config ----------------
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)


# ---------------- CSS + Dynamic Theme ----------------
def get_theme_css(color):
    return f"""
    <style>
    :root {{
        --theme-color: {color};
    }}

    h2, h3, h4, label {{
        color: var(--theme-color) !important;
    }}

    .bubble.user {{
        border-left: 3px solid var(--theme-color);
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

# Theme logic
theme_color = "#000000"  # neutral
if tool == "Bayut":
    theme_color = "#008060"
elif tool == "Dubizzle":
    theme_color = "#D92C27"

st.markdown(get_theme_css(theme_color), unsafe_allow_html=True)


# ---------------- Vectorstore Paths ----------------
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")


def find_best_matching_file(query):
    if not os.path.isdir(DATA_DIR):
        return None
    files = [f for f in os.listdir(DATA_DIR) if f != "faiss_store"]
    if not files:
        return None
    match = difflib.get_close_matches(query.lower(), [f.lower() for f in files], n=1, cutoff=0.3)
    for f in files:
        if f.lower() == match[0]:
            return os.path.join(DATA_DIR, f)
    return None


# ---------------- LangChain RAG ----------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class DummyLLM:
    def invoke(self, text):
        return text.split("ANSWER:", 1)[-1].strip() if "ANSWER:" in text else text


def get_local_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")


def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf": return PyPDFLoader(path).load()
        if ext == ".docx": return Docx2txtLoader(path).load()
        if ext in [".txt",".md"]: return TextLoader(path).load()
        if ext == ".csv": return CSVLoader(path).load()
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [Document(page_content=df.to_string(index=False))]
        return []
    except:
        return []


def build_vectorstore():
    docs = []
    for f in os.listdir(DATA_DIR):
        if f == "faiss_store": continue
        docs.extend(load_document(os.path.join(DATA_DIR, f)))
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50).split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


# ---------------- Load FAISS Index ----------------
vectorstore = None
if os.path.exists(INDEX_DIR):
    vectorstore = FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)
else:
    vectorstore = build_vectorstore()
    if vectorstore:
        vectorstore.save_local(INDEX_DIR)


# ---------------- Chat History ----------------
if "rag_history" not in st.session_state:
    st.session_state["rag_history"] = []


# ---------------- UI ----------------
st.write(f"### {tool}")

# Display previous chat
for q, a in st.session_state["rag_history"]:
    st.markdown(f"<div class='bubble user'>{q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{a}</div>", unsafe_allow_html=True)


# Input box
query = st.text_input("Ask your question:")


# ----------- BUTTONS (Horizontal Centered Layout) -----------
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
    st.success("Index cleared. Refresh page.")

if b2:
    st.session_state["rag_history"] = []

if b3:
    st.rerun()


# ---------------- Run Query ----------------
if query and vectorstore:

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    context = "\n".join([d.page_content[:1000] for d in docs]) if docs else ""

    if not context.strip():
        answer = "This information is not available in internal content."
    else:
        prompt = PromptTemplate.from_template("""
Answer using ONLY the context below.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt
            | get_local_llm()
            | StrOutputParser()
        )

        answer = chain.invoke({"question": query})

    st.session_state["rag_history"].append((query, answer))
    st.rerun()
