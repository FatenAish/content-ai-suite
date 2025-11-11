import os
import shutil
import pandas as pd
import streamlit as st
import difflib
import chardet  # helps detect encoding for CSV files

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="AI Content Lab – RAG Only", layout="wide")

# =========================================================
# CLEAN CSS
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
st.markdown("<h1 style='text-align:center;'>🔎 AI Content Lab – Internal RAG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Fast internal knowledge search using your uploaded documents.</p>", unsafe_allow_html=True)

# =========================================================
# FILE DOWNLOAD HELPER
# =========================================================
def find_best_matching_file(query, folder="data"):
    """Return the filename in /data that best matches the user's query."""
    if not os.path.isdir(folder):
        return None

    files = [f for f in os.listdir(folder) if f != "faiss_store"]
    if not files:
        return None

    best = difflib.get_close_matches(
        query.lower(),
        [f.lower() for f in files],
        n=1,
        cutoff=0.25
    )
    if best:
        for f in files:
            if f.lower() == best[0]:
                return os.path.join(folder, f)
    return None

# =========================================================
# RAG SYSTEM IMPORTS
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
try:
    # Newer versions
    from langchain_core.documents import Document
except Exception:
    # Older versions
    from langchain.schema import Document

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_local_llm():
    # Make sure GROQ_API_KEY is set in your environment
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

# =========================================================
# DOCUMENT LOADERS
# =========================================================
def load_document(path: str):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(path).load()

        if ext == ".docx":
            return Docx2txtLoader(path).load()

        if ext in [".txt", ".md"]:
            # autodetect enc for messy text files
            return TextLoader(path, autodetect_encoding=True).load()

        if ext == ".csv":
            # Detect encoding and fallback to utf-8 if unknown
            with open(path, "rb") as raw:
                detected = chardet.detect(raw.read())
            encoding = detected.get("encoding") or "utf-8"
            try:
                return CSVLoader(file_path=path, encoding=encoding, autodetect_encoding=True).load()
            except TypeError:
                # Older CSVLoader signature
                return CSVLoader(path, encoding=encoding).load()

        if ext == ".xlsx":
            df = pd.read_excel(path)
            # Convert to a single Document so the splitter can work
            content = df.to_string(index=False)
            return [Document(page_content=content, metadata={"source": path})]

    except Exception as e:
        print("Skipping bad file:", path, e)
        return []

    return []

def load_default_docs():
    docs = []
    if not os.path.isdir(DATA_DIR):
        return docs
    for f in os.listdir(DATA_DIR):
        if f == "faiss_store":
            continue
        p = os.path.join(DATA_DIR, f)
        if os.path.isfile(p):
            docs.extend(load_document(p))
    return docs

def faiss_exists():
    return os.path.isdir(INDEX_DIR)

def save_faiss(store: FAISS):
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
c1, c2 = st.columns([1, 1])

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
# LOAD INDEX
# =========================================================
if faiss_exists():
    with st.spinner("Loading FAISS index..."):
        vectorstore = load_faiss()
    st.success("✅ Index loaded")
else:
    docs = load_default_docs()
    if not docs:
        st.error("❌ No documents found in /data folder")
        st.stop()
    with st.spinner("Building index…"):
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
# USER QUERY
# =========================================================
query = st.text_input("Ask your question:")

if query:

    # ✅ Handle download requests first
    if any(x in query.lower() for x in ["download", "file", "get", "send", "share"]):
        match = find_best_matching_file(query)
        if match:
            st.success(f"✅ File matched: **{os.path.basename(match)}**")
            with open(match, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {os.path.basename(match)}",
                    data=f.read(),  # bytes, not file object
                    file_name=os.path.basename(match),
                    mime="application/octet-stream"
                )
            st.stop()
        else:
            st.error("❌ No matching file found.")
            st.stop()

    with st.spinner("Thinking…"):

        # Evidence search (for preview)
        hits = vectorstore.similarity_search_with_score(str(query), k=3)

        # Stable retriever
        rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Prepare context
        prepare_context = RunnableLambda(
            lambda docs: "\n\n".join(getattr(d, "page_content", str(d))[:2000] for d in docs)
        )

        # ✅ Clean prompt
        prompt = PromptTemplate.from_template("""
You are an internal AI assistant for Dubizzle Group.
You must ALWAYS give:

✅ Clear
✅ Detailed
✅ Structured
✅ Professional
✅ Helpful
✅ Human-like answers

Use ONLY the context below to answer the question.

If the context contains the answer:
- Give a rich, complete explanation
- Add examples if helpful
- Organize the answer in sections with headers

If the answer is partially in the context:
- Combine what's available
- Fill missing logic carefully

If the context does NOT contain the answer:
- Say: “The internal documents do not contain this information.”
- THEN give a general helpful answer using your own knowledge.

NEVER give short or shallow answers.

=====================================
CONTEXT:
{context}
=====================================

QUESTION:
{question}

=====================================

DETAILED ANSWER:
""")

        # ✅ Minimal, correct chain: input is the question string
        chain = (
            {
                "context": (rag_retriever | prepare_context),
                "question": RunnablePassthrough()
            }
            | prompt
            | get_local_llm()
            | StrOutputParser()
        )

        answer = chain.invoke(query)

        st.session_state["rag_history"].append((query, answer))
        st.markdown(f"<div class='bubble user'>{query}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bubble ai'>{answer}</div>", unsafe_allow_html=True)

        # Evidence display
        if hits:
            st.markdown("### 📎 Evidence")
            for i, (doc, score) in enumerate(hits, 1):
                snippet = getattr(doc, "page_content", str(doc))[:350]
                st.markdown(
                    f"<div class='evidence'><b>{i}.</b> similarity={score:.3f}<br>{snippet}…</div>",
                    unsafe_allow_html=True
                )
