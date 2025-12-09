import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)

# -----------------------------
# SIDEBAR MENU
# -----------------------------
with st.sidebar:
    st.header("Select an option")
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])

# -----------------------------
# HEADER AND SUBTITLE
# -----------------------------
st.markdown("""
    <h1 style='font-size:42px;'>
        <span style='color:#0E8A6D;'>Bayut</span> & 
        <span style='color:#D71920;'>Dubizzle</span> 
        AI Content Assistant
    </h1>
    <p style='font-size:18px; color:#444; margin-top:-10px;'>
        Fast internal knowledge search powered by internal content.
    </p>
""", unsafe_allow_html=True)

# -----------------------------
# DATA DIRECTORY (DISPLAY ONLY)
# -----------------------------
DATA_DIR = "./data"
st.markdown(f"### 📁 DATA DIR: `/app/data`")

# INTERNAL USE ONLY – DO NOT DISPLAY FILE LIST
files = os.listdir(DATA_DIR)

# -----------------------------
# VECTOR STORE PATHS
# -----------------------------
FAISS_DIR = os.path.join(DATA_DIR, "faiss_store")

# -----------------------------
# EMBEDDINGS MODEL
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# LOAD / BUILD INDEX
# -----------------------------
def load_vectorstore():
    """Load FAISS if exists, otherwise return None."""
    if os.path.exists(FAISS_DIR):
        try:
            return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        except:
            return None
    return None

def build_vectorstore():
    """Read all .txt files, embed them, and save FAISS."""
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_DIR, file), encoding="utf-8")
            pages = loader.load()
            documents.extend(splitter.split_documents(pages))

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_DIR)
    return db

# Load or create vectorstore
vectorstore = load_vectorstore()
if vectorstore is None:
    vectorstore = build_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# USER INPUT FIELD
# -----------------------------
st.subheader(mode)
query = st.text_input("Ask your question:")

# -----------------------------
# PROCESS QUERY
# -----------------------------
if query:
    with st.spinner("Searching internal knowledge..."):
        time.sleep(0.2)

        # Retrieve relevant text chunks
        docs = retriever.get_relevant_documents(query)

        if not docs:
            st.write("I couldn't find anything related to this question in the internal documents.")
        else:
            st.markdown("### Here’s what I found:")
            for i, d in enumerate(docs, start=1):
                st.markdown(f"**Source:** {d.metadata.get('source', 'Unknown File')}")
                st.write(d.page_content)
                st.markdown("---")

# -----------------------------
# REBUILD INDEX BUTTON
# -----------------------------
if st.button("Rebuild Index"):
    with st.spinner("Rebuilding vector index..."):
        vectorstore = build_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.success("Index rebuilt successfully!")
