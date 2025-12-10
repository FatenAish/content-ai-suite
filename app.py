import streamlit as st
import os
import time

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <h4 style="margin-top: 30px;">Select an option</h4>
        """,
        unsafe_allow_html=True
    )
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])


# ============================================================
# HEADER (EXACT DESIGN)
# ============================================================
st.markdown(
    """
    <div style="text-align:center; margin-top:-30px;">
        <h1 style="font-size:42px; font-weight:700;">
            <span style="color:#0E8A6D;">Bayut</span> 
            & 
            <span style="color:#D71920;">Dubizzle</span>
            AI Content Assistant
        </h1>
        <p style="font-size:16px; color:#444; margin-top:-15px;">
            Fast internal knowledge search powered by internal content.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA DIR (EXACT LIKE SCREENSHOT)
# ============================================================
DATA_DIR = "/app/data"
LOCAL_FALLBACK = "./data"

if os.path.exists(LOCAL_FALLBACK):
    DATA_DIR = LOCAL_FALLBACK

st.markdown(
    f"""
    <div style="font-size:18px; margin-top:20px;">
        📁 <strong>DATA DIR:</strong> {DATA_DIR}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EMBEDDINGS
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# VECTORSTORE PATH
# ============================================================
FAISS_DIR = os.path.join(DATA_DIR, "faiss_store")


# ============================================================
# SAFE VECTORSTORE BUILDER (NO CRASHING)
# ============================================================
def build_vectorstore():
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(DATA_DIR, file)

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            if not text.strip():
                st.warning(f"⚠️ Skipping empty file: {file}")
                continue

            splits = splitter.split_text(text)
            docs = [Document(page_content=s, metadata={"source": file}) for s in splits]
            documents.extend(docs)

        except Exception as e:
            st.error(f"❌ Error reading file `{file}` — {e}")
            continue

    if not documents:
        st.error("❌ No readable text files found. Cannot create index.")
        return None

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_DIR)
    return db


# ============================================================
# LOAD VECTORSTORE (SAFE)
# ============================================================
def load_vectorstore():
    if os.path.exists(FAISS_DIR):
        try:
            return FAISS.load_local(
                FAISS_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"❌ Could not load FAISS index: {e}")
            return None
    return None


# ============================================================
# INIT VECTORSTORE
# ============================================================
vectorstore = load_vectorstore()
if vectorstore is None:
    vectorstore = build_vectorstore()

if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
else:
    retriever = None


# ============================================================
# SEARCH UI
# ============================================================
st.subheader(mode)
query = st.text_input("Ask your question:")


# ============================================================
# PERFORM SEARCH
# ============================================================
if query and retriever:
    with st.spinner("Searching internal knowledge..."):
        time.sleep(0.3)

        docs = retriever.invoke(query)

        if not docs:
            st.write("No matching internal information found.")
        else:
            st.markdown("### 🔎 Results Found:")
            for d in docs:
                st.markdown(f"**📄 Source:** {d.metadata.get('source', 'Unknown')}")
                st.write(d.page_content)
                st.markdown("---")


# ============================================================
# REBUILD BUTTON
# ============================================================
if st.button("Rebuild Index"):
    with st.spinner("Rebuilding vector index..."):
        vectorstore = build_vectorstore()
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            st.success("Index rebuilt successfully!")
        else:
            st.error("Index could not be built.")
