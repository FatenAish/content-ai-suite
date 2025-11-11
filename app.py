import os
import shutil
import pandas as pd
import streamlit as st
import difflib

# ==============================================
# PAGE CONFIG
# ==============================================
st.set_page_config(page_title="AI Content Lab – RAG Only", layout="wide")

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
  line-height: 1.5;
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

.header { text-align:center; width:100%; margin-bottom:20px; }
.header-title { font-size:32px; font-weight:900; }
.header-title .red { color:#D92C27; }
.header-sub { font-size:15px; color:#4b5563; }
</style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.markdown('<div class="header">', unsafe_allow_html=True)
st.markdown('<div class="header-title"><span class="red">AI</span> Content Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Internal Knowledge Base (Local RAG)</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# ✅✅✅ INTERNAL RAG ENGINE ONLY
# =====================================================================

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


DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")


# ✅ Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ✅ LLM
def get_local_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)


# =====================================
# LOAD DOCUMENTS
# =====================================
def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        if ext == ".docx":
            return Docx2txtLoader(path).load()
        if ext in [".txt", ".md"]:
            return TextLoader(path).load()
        if ext == ".csv":
            return CSVLoader(path, encoding="utf-8").load()
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [{"page_content": df.to_string(), "metadata": {"source": path}}]
    except:
        return []
    return []


def load_all_docs():
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


# =====================================
# FAISS INDEX FUNCTIONS
# =====================================
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
def build_index(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    return FAISS.from_texts(texts, get_embeddings())


# =====================================
# ✅ FILE SEARCH + DOWNLOAD
# =====================================

def find_matching_file(user_input):
    """Find best-matching file in /data."""
    try:
        files = [
            f for f in os.listdir(DATA_DIR)
            if os.path.isfile(os.path.join(DATA_DIR, f))
        ]

        best = difflib.get_close_matches(
            user_input.lower(),
            [f.lower() for f in files],
            n=1,
            cutoff=0.4
        )

        if best:
            for f in files:
                if f.lower() == best[0]:
                    return f, files

        return None, files

    except Exception:
        return None, []


# =====================================
# BUTTONS
# =====================================
col1, col2 = st.columns(2)

if col1.button("🔄 Rebuild Index"):
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()
    st.success("Index cleared. Refresh the app.")
    st.stop()

if col2.button("🧹 Clear Chat"):
    st.session_state.pop("history", None)
    st.rerun()


# =====================================
# LOAD OR BUILD INDEX
# =====================================
if faiss_exists():
    with st.spinner("Loading FAISS index..."):
        vectorstore = load_faiss()
else:
    docs = load_all_docs()
    if not docs:
        st.error("❌ No documents found in /data folder.")
        st.stop()
    with st.spinner("Indexing documents..."):
        vectorstore = build_index(docs)
        save_faiss(vectorstore)
    st.success("✅ Index created")


# =====================================
# CHAT HISTORY
# =====================================
if "history" not in st.session_state:
    st.session_state["history"] = []

for q, a in st.session_state["history"]:
    st.markdown(f"<div class='bubble user'>{q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{a}</div>", unsafe_allow_html=True)


# =====================================
# ✅ USER INPUT
# =====================================
query = st.text_input("Ask your question:")


if query:

    q_lower = query.lower()

    # ✅ Detect downloads
    download_terms = ["download", "file", "pdf", "doc", "send", "give me"]

    if any(t in q_lower for t in download_terms):
        matched_file, files = find_matching_file(query)

        if matched_file:
            fpath = os.path.join(DATA_DIR, matched_file)

            st.markdown(f"<div class='bubble user'>{query}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='bubble ai'>✅ Found <b>{matched_file}</b><br>Click below to download.</div>",
                unsafe_allow_html=True
            )

            with open(fpath, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {matched_file}",
                    data=f,
                    file_name=matched_file,
                    mime="application/octet-stream"
                )

            st.stop()

        else:
            st.markdown(f"<div class='bubble ai'>❌ No matching file found.<br><br>"
                        f"Available files:<br>{'<br>'.join(files)}</div>",
                        unsafe_allow_html=True)
            st.stop()

    # ✅ Otherwise: DETAILED RAG ANSWER
    with st.spinner("Thinking..."):

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        hits = vectorstore.similarity_search_with_score(query, k=3)

        extract_question = RunnableLambda(lambda x: x["question"])
        retrieve_docs = extract_question | retriever
        prepare_ctx = RunnableLambda(
            lambda docs: "\n\n".join(d.page_content[:2000] for d in docs)
        )

        ctx_pipe = retrieve_docs | prepare_ctx

        prompt = PromptTemplate.from_template(
            "Answer in detailed paragraphs using ONLY this context.\n"
            "If the context does not contain an answer, say so politely.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "DETAILED ANSWER:"
        )

        chain = (
            {"context": ctx_pipe, "question": RunnablePassthrough()}
            | prompt
            | get_local_llm()
            | StrOutputParser()
        )

        answer = chain.invoke({"question": query})

    # Save and show response
    st.session_state["history"].append((query, answer))
    st.markdown(f"<div class='bubble user'>{query}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{answer}</div>", unsafe_allow_html=True)

    # ✅ Evidence
    if hits:
        st.write("### 📎 Evidence")
        for i, (doc, score) in enumerate(hits, 1):
            snippet = doc.page_content[:350]
            st.markdown(
                f"<div class='evidence'><b>{i}.</b> similarity={score:.3f}<br>{snippet}...</div>",
                unsafe_allow_html=True
            )
