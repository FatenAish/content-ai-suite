import os
import shutil
import pandas as pd
import streamlit as st
import difflib
import chardet
from langchain_core.documents import Document  # ✅ Needed for XLSX loader

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Dubizzle Group AI Content Lab – Internal RAG", layout="wide")

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
.title-red {
  font-size: 34px;
  font-weight: 900;
  color: #D92C27;
  text-align:center;
}
.title-main {
  font-size: 28px;
  font-weight: 700;
  color: #111;
  text-align:center;
  margin-top: -10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='title-red'>Dubizzle Group AI Lab</div>", unsafe_allow_html=True)
st.markdown("<div class='title-main'>Internal RAG</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Fast internal knowledge search using your uploaded documents.</p>", unsafe_allow_html=True)

# =========================================================
# FILE DOWNLOAD HELPER
# =========================================================
def find_best_matching_file(query, folder="data"):
    if not os.path.isdir(folder):
        return None

    files = [f for f in os.listdir(folder) if f != "faiss_store"]
    if not files:
        return None

    match = difflib.get_close_matches(query.lower(), [f.lower() for f in files], n=1, cutoff=0.3)
    if not match:
        return None

    for f in files:
        if f.lower() == match[0]:
            return os.path.join(folder, f)
    return None

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

def get_local_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

# =========================================================
# DOCUMENT LOADERS
# =========================================================
def load_document(path: str):
    ext = os.path.splitext(path)[1].lower()

    try:
        # PDF
        if ext == ".pdf":
            docs = PyPDFLoader(path).load()
            return docs

        # DOCX
        if ext == ".docx":
            docs = Docx2txtLoader(path).load()
            return docs

        # TXT / MD
        if ext in [".txt", ".md"]:
            docs = TextLoader(path, autodetect_encoding=True).load()
            return docs

        # CSV (try multiple encodings)
        if ext == ".csv":
            for enc in ["utf-8", "utf-8-sig", "cp1256", "windows-1256", None]:
                try:
                    docs = CSVLoader(path, encoding=enc).load()
                    return docs
                except:
                    continue
            return []

        # XLSX
        if ext == ".xlsx":
            df = pd.read_excel(path)
            content = df.to_string(index=False)
            return [Document(page_content=content, metadata={"source": path})]

        return []

    except Exception as e:
        print("Skipping file:", path, e)
        return []

def load_default_docs():
    docs = []
    if not os.path.isdir(DATA_DIR):
        return docs

    for f in os.listdir(DATA_DIR):
        if f == "faiss_store":
            continue
        full = os.path.join(DATA_DIR, f)
        if os.path.isfile(full):
            docs.extend(load_document(full))

    return docs

def faiss_exists():
    return os.path.isdir(INDEX_DIR)

@st.cache_resource
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    return FAISS.from_texts(texts, get_embeddings())

def save_faiss(store):
    store.save_local(INDEX_DIR)

def load_faiss():
    return FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)

# =========================================================
# LOAD VECTORSTORE
# =========================================================
if faiss_exists():
    vectorstore = load_faiss()
    st.success("✅ Index loaded")
else:
    docs = load_default_docs()
    if not docs:
        st.error("❌ No documents found in /data folder.")
        st.stop()

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
# USER QUESTION
# =========================================================
query = st.text_input("Ask your question:")

# =========================================================
# ✅ CENTERED TOOL BUTTONS BELOW QUESTION
# =========================================================
st.write("")  # small spacing
left, mid, right = st.columns([1,1,1])

with mid:
    b1 = st.button("🔄 Rebuild Index")
    b2 = st.button("🧹 Clear Chat")

# Actions
if b1:
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()
    st.success("✅ Index cleared. Refresh the page.")
    st.stop()

if b2:
    st.session_state.pop("rag_history", None)
    st.rerun()

# =========================================================
# ANSWERING
# =========================================================
if query:

    # -------- File download intent ------
    if any(x in query.lower() for x in ["download", "file", "get", "send", "share"]):
        match = find_best_matching_file(query)
        if match:
            st.success(f"✅ File ready: **{os.path.basename(match)}**")

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

    # -------- RAG answering --------
    with st.spinner("Thinking…"):

        hits = vectorstore.similarity_search_with_score(query, k=3)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        extract_q = RunnableLambda(lambda x: x["question"])
        retrieve_docs = extract_q | retriever

        prepare_context = RunnableLambda(
            lambda docs: "\n\n".join(d.page_content[:1800] for d in docs)
        )

        context_pipeline = retrieve_docs | prepare_context

        prompt = PromptTemplate.from_template("""
You are an internal AI assistant for Dubizzle Group.

✅ Always clear  
✅ Always structured  
✅ Always detailed  
✅ Use context first  
✅ If missing info, fill logically  

==========================
CONTEXT:
{context}
==========================

QUESTION:
{question}

==========================
DETAILED ANSWER:
""")

        chain = (
            {"context": context_pipeline, "question": RunnablePassthrough()}
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
