import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ------------------------------
# UI SETUP
# ------------------------------
st.set_page_config(page_title="Bayut & Dubizzle AI Assistant", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#008060;'>Bayut <span style='color:#d30000'>& Dubizzle</span> AI Content Assistant</h1>
<p style='text-align:center;'>Fast internal knowledge search powered by internal content.</p>
""", unsafe_allow_html=True)


# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------
# LOAD EMBEDDINGS
# ------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

emb = load_embeddings()

# ------------------------------
# LOAD DOCUMENTS FROM /data
# ------------------------------
def load_docs():
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)
            text = open(path, "r", encoding="utf-8", errors="ignore").read()
            docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs

# ------------------------------
# BUILD VECTORSTORE
# ------------------------------
@st.cache_resource
def build_vectorstore():
    docs = load_docs()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, emb)

vectorstore = build_vectorstore()

# ------------------------------
# GET RETRIEVER (IMPORTANT FIX)
# ------------------------------
retriever = None
if vectorstore:
    retriever = vectorstore.as_retriever()

# ------------------------------
# SIMPLE LOCAL LLM (NO GROQ)
# ------------------------------
# To avoid import errors we use a dummy offline LLM-like function
def local_llm(prompt, documents):
    text = "\n\n".join([d.page_content[:500] for d in documents])
    return f"🔍 **Answer from retrieved documents**\n\n{text[:1500]}"

# ------------------------------
# SHOW FILES
# ------------------------------
st.subheader("📁 DATA DIR:", divider="gray")
st.write(f"Location: `{DATA_DIR}`")

files_list = os.listdir(DATA_DIR)
st.write(files_list)

st.write("---")

# ------------------------------
# USER INPUT
# ------------------------------
st.subheader("General")
query = st.text_input("Ask your question:")

col1, col2, col3 = st.columns(3)
with col1:
    rebuild = st.button("Rebuild Index")
with col2:
    st.button("Clear Chat")
with col3:
    st.button("Reload")

# ------------------------------
# REBUILD INDEX
# ------------------------------
if rebuild:
    st.cache_resource.clear()
    st.success("Index rebuilt. Refreshing...")
    st.stop()

# ------------------------------
# PROCESS QUERY
# ------------------------------
if query:
    if not retriever:
        st.error("No vectorstore found. Upload files to /data and rebuild index.")
        st.stop()

    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        st.error(f"Retriever error: {str(e)}")
        st.stop()

    if not docs:
        st.warning("No relevant information found in documents.")
        st.stop()

    answer = local_llm(query, docs)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Documents")
    for d in docs:
        st.write(f"📄 **{d.metadata['source']}**")
        st.write(d.page_content[:500] + "...")
