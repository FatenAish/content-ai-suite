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
# Load documents from /data (.txt only – matches your repo)
# ---------------------------------------------------------
def load_documents_from_data() -> list[Document]:
    docs: list[Document] = []

    if not os.path.isdir(DATA_DIR):
        return docs

    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".txt"):
            continue

        fpath = os.path.join(DATA_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": fname}))
        except Exception:
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
        # no history now, just rerun to clear output
        st.rerun()
with col3:
    if st.button("Reload"):
        st.rerun()


# ---------------------------------------------------------
# Simple “LLM”: just formats the single best chunk
# ---------------------------------------------------------
def simple_answer_from_docs(question: str, docs: list[Document]) -> str:
    """Create a short answer using the most relevant chunk only."""
    if not docs:
        return "I couldn't find anything related to this question in the internal documents."

    best = docs[0]
    snippet = best.page_content[:1200]  # limit length a bit

    answer = (
        f"Here’s what I found in the internal content related to:\n"
        f"**{question}**\n\n"
        f"From **{best.metadata.get('source', 'unknown')}**:\n\n"
        f"{snippet}"
    )
    return answer


# ---------------------------------------------------------
# Run query – return ONE answer only, no history
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

            # New LangChain retrievers are Runnables – use invoke()
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                # Fallback for older versions
                docs = retriever.get_relevant_documents(query)

            answer = simple_answer_from_docs(query, docs)

            # show one “user bubble”
            st.markdown(
                f"<div style='background:#f5f5f5;padding:10px 14px;border-radius:6px;"
                f"margin:4px 0;border-left:3px solid #008060;'>"
                f"{query}</div>",
                unsafe_allow_html=True,
            )
            # and one “assistant bubble”
            st.markdown(
                f"<div style='background:#ffffff;padding:10px 14px;border-radius:6px;"
                f"margin:4px 0;border:1px solid #eee;'>"
                f"{answer}</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Retriever error: {e}")
