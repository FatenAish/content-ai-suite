# =========================================================
# Bayut & Dubizzle AI Content Assistant — Simple Internal RAG
# =========================================================

import os
import json
import streamlit as st

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# Page config & basic styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide",
)

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

# ---------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------
DATA_DIR = "data"  # works locally and on Streamlit Cloud
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Embeddings (cached)
# ---------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------
# Load documents from /data
# (only .txt files for now – matches your repo)
# ---------------------------------------------------------
def load_documents_from_data() -> list[Document]:
    docs: list[Document] = []

    if not os.path.isdir(DATA_DIR):
        return docs

    for fname in os.listdir(DATA_DIR):
        # only text files with content
        if not fname.lower().endswith(".txt"):
            continue

        fpath = os.path.join(DATA_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": fname}))
        except Exception:
            # skip any problematic file but don't crash the app
            continue

    return docs


# ---------------------------------------------------------
# Build FAISS vectorstore in-memory (cached)
# ---------------------------------------------------------
@st.cache_resource
def build_vectorstore():
    docs = load_documents_from_data()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


vectorstore = build_vectorstore()

# ---------------------------------------------------------
# Helper to rebuild index (clear cache)
# ---------------------------------------------------------
def rebuild_index():
    st.cache_resource.clear()
    st.rerun()


# ---------------------------------------------------------
# Show which files are loaded
# ---------------------------------------------------------
st.caption(f"📁 DATA DIR: `{DATA_DIR}`")

if os.path.isdir(DATA_DIR):
    st.write("Files found in `data/`:")
    st.json(os.listdir(DATA_DIR))
else:
    st.warning("`data/` folder not found. Please create it and add .txt files.")

st.write("---")

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.subheader("General")

query = st.text_input("Ask your question:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Rebuild Index"):
        rebuild_index()
with col2:
    if st.button("Clear Chat"):
        if "history" in st.session_state:
            st.session_state["history"] = []
        st.rerun()
with col3:
    if st.button("Reload"):
        st.rerun()

# Initialise simple chat history
if "history" not in st.session_state:
    st.session_state["history"] = []


# ---------------------------------------------------------
# Simple local "LLM" – just summarises retrieved docs
# (no external API needed)
# ---------------------------------------------------------
def simple_answer_from_docs(question: str, docs: list[Document]) -> str:
    """Create a human-readable answer using the retrieved chunks."""
    if not docs:
        return "I couldn't find anything related to this question in the internal documents."

    # Use the most relevant 2–3 chunks
    top_chunks = docs[:3]
    combined_text = "\n\n---\n\n".join(
        f"From **{d.metadata.get('source', 'unknown')}**:\n\n{d.page_content[:1000]}"
        for d in top_chunks
    )

    answer = (
        f"Here’s what I found in the internal content related to:\n\n"
        f"**{question}**\n\n"
        f"{combined_text}"
    )
    return answer


# ---------------------------------------------------------
# Run query
# ---------------------------------------------------------
if query.strip():
    if vectorstore is None:
        st.error(
            "No documents found to build the index. "
            "Please add .txt files to the `data/` folder and click **Rebuild Index**."
        )
    else:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(query)
        except Exception as e:
            st.error(f"Retriever error: {e}")
            docs = []

        answer = simple_answer_from_docs(query, docs)

        st.session_state["history"].append({"q": query, "a": answer})

# ---------------------------------------------------------
# Show chat history
# ---------------------------------------------------------
for item in reversed(st.session_state["history"]):
    st.markdown(
        f"<div style='background:#f5f5f5;padding:10px 14px;border-radius:6px;"
        f"margin:4px 0;border-left:3px solid #008060;'>"
        f"{item['q']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:#ffffff;padding:10px 14px;border-radius:6px;"
        f"margin:4px 0;border:1px solid #eee;'>"
        f"{item['a']}</div>",
        unsafe_allow_html=True,
    )
