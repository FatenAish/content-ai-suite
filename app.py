import streamlit as st
import os
import time

# Correct new imports after LangChain split
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.header("Select an option")
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("""
    <h1 style='font-size:42px; margin-bottom: 0px;'>
        <span style='color:#0E8A6D;'>Bayut</span> & 
        <span style='color:#D71920;'>Dubizzle</span>
        AI Content Assistant
    </h1>
    <p style='font-size:18px; color:#444; margin-top:-8px;'>
        Fast internal knowledge search powered by internal content.
    </p>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# DATA DIRECTORY
# -------------------------------------------------------
DATA_DIR = "./data"
FAISS_DIR = os.path.join(DATA_DIR, "faiss_store")

st.markdown("### 📁 DATA DIR: `/app/data`")


# -------------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------------------------------
# LOAD / BUILD INDEX
# -------------------------------------------------------
def load_vectorstore():
    """Load existing FAISS index."""
    if os.path.exists(FAISS_DIR):
        try:
            return FAISS.load_local(
                FAISS_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            return None
    return None


def build_vectorstore():
    """Embed all TXT files and save FAISS index."""
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_DIR, file), encoding="utf-8")
            pages = loader.load()
            documents.extend(splitter.split_documents(pages))

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_DIR)
    return db


vectorstore = load_vectorstore()
if vectorstore is None:
    vectorstore = build_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -------------------------------------------------------
# INPUT
# -------------------------------------------------------
st.subheader(mode)
query = st.text_input("Ask your question:")


# -------------------------------------------------------
# SEARCH
# -------------------------------------------------------
if query:
    with st.spinner("Searching internal knowledge..."):
        time.sleep(0.2)

        docs = retriever.get_relevant_documents(query)

        if not docs:
            st.info("I couldn't find anything related to this question in the internal documents.")
        else:
            st.markdown("### Here’s what I found:")
            for d in docs:
                src = d.metadata.get("source", "Unknown file")
                st.markdown(f"**From {src}:**")
                st.write(d.page_content)
                st.markdown("---")


# -------------------------------------------------------
# REBUILD INDEX
# -------------------------------------------------------
if st.button("Rebuild Index"):
    with st.spinner("Rebuilding index..."):
        vectorstore = build_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    st.success("Index rebuilt successfully!")
