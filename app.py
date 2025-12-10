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
# HEADER (exact design)
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
# DATA DIR (NOT DISPLAYED)
# ============================================================
DATA_DIR = "/app/data"
LOCAL_FALLBACK = "./data"
if os.path.exists(LOCAL_FALLBACK):
    DATA_DIR = LOCAL_FALLBACK


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
# BUILD VECTORSTORE (safe)
# ============================================================
def build_vectorstore():
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            try:
                with open(
                    os.path.join(DATA_DIR, file),
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:
                    text = f.read()

                if not text.strip():
                    continue

                chunks = splitter.split_text(text)
                for c in chunks:
                    documents.append(
                        Document(page_content=c, metadata={"source": file})
                    )

            except:
                continue

    if not documents:
        return None

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_DIR)
    return db


# ============================================================
# LOAD VECTORSTORE
# ============================================================
def load_vectorstore():
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


# ============================================================
# INIT VECTORSTORE
# ============================================================
vectorstore = load_vectorstore()
if vectorstore is None:
    vectorstore = build_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # ONE MATCH


# ============================================================
# INPUT
# ============================================================
st.subheader(mode)
query = st.text_input("Ask your question:")


# ============================================================
# CLEAN ANSWER EXTRACTION LOGIC
# ============================================================
def extract_clean_answer(raw_text, user_question):

    # Normalize
    raw = raw_text.strip()
    user_q = user_question.lower().strip()

    # Split the text into sections based on Q-number formatting
    parts = raw.split("Q")

    best_section = ""

    for p in parts:
        section = p.strip()
        if not section:
            continue

        # Check if this section contains the question text
        if user_q in section.lower():
            best_section = section
            break

    # If nothing matches, return raw text
    if not best_section:
        return raw

    # Remove "4 –" or similar
    if "–" in best_section:
        best_section = best_section.split("–", 1)[1].strip()

    return best_section


# ============================================================
# SEARCH – SHOW ONLY THE EXACT ANSWER
# ============================================================
if query:
    with st.spinner("Searching internal knowledge..."):
        docs = retriever.invoke(query)

        if not docs:
            st.write("No matching internal information found.")
        else:
            best = docs[0]

            clean_answer = extract_clean_answer(best.page_content, query)

            st.markdown("### ✅ Best Match")
            st.write(clean_answer)

            st.markdown("---")
            st.markdown("### 📄 Source")
            st.write(f"File: {best.metadata.get('source')}")


# ============================================================
# REBUILD INDEX
# ============================================================
if st.button("Rebuild Index"):
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    st.success("Index rebuilt successfully!")
