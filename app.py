import os
import shutil
import difflib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Dubizzle Group AI Content Lab – Internal RAG", layout="wide")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
[data-testid="stVerticalBlock"] > div {
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
main .block-container { padding-top: 0rem !important; }

.bubble {
  padding: 12px 16px;
  border-radius: 14px;
  margin: 6px 0;
  max-width: 85%;
  line-height: 1.6;
  font-size: 15px;
}
.bubble.user { background: #f2f2f2; margin-left: auto; }
.bubble.ai { background: #ffffff; margin-right: auto; }

.header-title {
  font-size: 36px;
  font-weight: 900;
  text-align: center;
  color: #111;
}
.header-title .red { color: #D92C27; }   /* Dubizzle Red */
.header-title .green { color: #1DBF73; } /* Bayut Green */

.sub {
  text-align:center;
  font-size:15px;
  color:#555;
}

.evidence {
  background:#fafafa;
  border:1px solid #e5e7eb;
  border-radius:12px;
  padding:10px;
  margin:10px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-top:10px;">
    <span style="font-size:36px; font-weight:900; color:#D92C27;">
        Dubizzle Group AI Lab
    </span>
    <span style="font-size:36px; font-weight:900; color:#111;">
        – Internal RAG
    </span>
</div>

<div style="text-align:center; font-size:15px; color:#555;">
    Internal AI-powered knowledge system for Bayut & Dubizzle teams
</div>
""", unsafe_allow_html=True)


# =========================================================
# RAG IMPORTS
# =========================================================
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load Groq API Key from Streamlit Secrets
def get_local_llm():
    api_key = st.secrets["GROQ_API_KEY"]
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=api_key
    )

# =========================================================
# DOCUMENT LOADING
# =========================================================
def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf": return PyPDFLoader(path).load()
        if ext == ".docx": return Docx2txtLoader(path).load()
        if ext in [".txt", ".md"]: return TextLoader(path).load()
        if ext == ".csv": return CSVLoader(path, encoding="utf-8").load()
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [{"page_content": df.to_string(), "metadata": {"source": path}}]
    except:
        return []

def load_default_docs():
    docs = []
    if not os.path.isdir(DATA_DIR): return docs

    for f in os.listdir(DATA_DIR):
        if f == "faiss_store":
            continue
        p = os.path.join(DATA_DIR, f)
        if os.path.isfile(p):
            docs.extend(load_document(p))
    return docs

def faiss_exists():
    return os.path.isdir(INDEX_DIR)

def save_faiss(store):
    os.makedirs(DATA_DIR, exist_ok=True)
    store.save_local(INDEX_DIR)

def load_faiss():
    return FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    return FAISS.from_texts(texts, get_embeddings())

# =========================================================
# BUTTONS
# =========================================================
c1, c2 = st.columns(2)

if c1.button("🔄 Rebuild Index"):
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()
    st.success("Index cleared. Restart app to rebuild.")
    st.stop()

if c2.button("🧹 Clear Chat"):
    st.session_state.pop("rag_history", None)
    st.rerun()

# =========================================================
# LOAD VECTORSTORE
# =========================================================
if faiss_exists():
    with st.spinner("Loading knowledge base…"):
        vectorstore = load_faiss()
    st.success("✅ Index loaded")
else:
    docs = load_default_docs()
    if not docs:
        st.error("❌ No documents found in /data")
        st.stop()

    with st.spinner("Creating index…"):
        vectorstore = build_vectorstore(docs)
        save_faiss(vectorstore)

    st.success("✅ Index created")

# =========================================================
# CHAT HISTORY
# =========================================================
if "rag_history" not in st.session_state:
    st.session_state["rag_history"] = []

for q, a in st.session_state["rag_history"]:
    st.markdown(f"<div class='bubble user'>{q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{a}</div>", unsafe_allow_html=True)

# =========================================================
# USER INPUT
# =========================================================
query = st.text_input("Ask your question:")

if query:

    # ✅ Check for download requests
    if any(x in query.lower() for x in ["download", "file", "get", "send", "share"]):
        match = find_best_matching_file(query)
        if match:
            st.success(f"✅ File found: **{os.path.basename(match)}**")
            with open(match, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {os.path.basename(match)}",
                    data=f,
                    file_name=os.path.basename(match),
                    mime="application/octet-stream"
                )
            st.stop()
        else:
            st.error("❌ No matching file found.")
            st.stop()

    with st.spinner("Thinking…"):

        hits = vectorstore.similarity_search_with_score(str(query), k=3)
        rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        extract_q = RunnableLambda(lambda x: x["question"])
        retrieve_docs = extract_q | rag_retriever

        prepare_context = RunnableLambda(
            lambda docs: "\n\n".join(d.page_content[:1500] for d in docs)
        )

        context_pipeline = retrieve_docs | prepare_context

        prompt = PromptTemplate.from_template("""
You are the official AI assistant for Dubizzle Group.

✅ Always answer in:
- Clear structure
- Full detail
- Professional tone
- Step-by-step logic
- Helpful and actionable advice

Use ONLY the provided context. If something is missing, still give useful reasoning.

====================
CONTEXT:
{context}
====================

QUESTION:
{question}

====================
DETAILED ANSWER:
""")

        chain = (
            {
                "context": context_pipeline,
                "question": RunnablePassthrough(),
            }
            | prompt
            | get_local_llm()
            | StrOutputParser()
        )

        answer = chain.invoke({"question": query})

        st.session_state["rag_history"].append((query, answer))

        st.markdown(f"<div class='bubble user'>{query}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bubble ai'>{answer}</div>", unsafe_allow_html=True)

        if hits:
            st.markdown("### 📎 Evidence")
            for i, (doc, score) in enumerate(hits, 1):
                snippet = doc.page_content[:350]
                st.markdown(
                    f"<div class='evidence'><b>{i}.</b> similarity={score:.3f}<br>{snippet}…</div>",
                    unsafe_allow_html=True
                )
