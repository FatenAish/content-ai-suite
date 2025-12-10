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
# CHAT HISTORY STATE (INVISIBLE)
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []


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
# HEADER
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
# DATA DIR (HIDDEN)
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
# BUILD VECTORSTORE (SAFE)
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
                    documents.append(Document(page_content=c, metadata={"source": file}))

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
# CLEAN ANSWER EXTRACTION
# ============================================================
def extract_clean_answer(raw_text, user_question):

    raw = raw_text.strip()
    user_q = user_question.lower().strip()

    parts = raw.split("Q")
    best_section = ""

    for p in parts:
        section = p.strip()
        if not section:
            continue

        if user_q in section.lower():
            best_section = section
            break

    if not best_section:
        return raw

    if "–" in best_section:
        best_section = best_section.split("–", 1)[1].strip()

    return best_section


# ============================================================
# INPUT FIELD
# ============================================================
st.subheader(mode)
query = st.text_input("Ask your question:")


# ============================================================
# SEARCH + SAVE TO CHAT FEED
# ============================================================
if query:
    with st.spinner("Searching internal knowledge..."):
        docs = retriever.invoke(query)

        if docs:
            best = docs[0]
            clean_answer = extract_clean_answer(best.page_content, query)

            st.session_state.history.append({
                "question": query,
                "answer": clean_answer
            })
        else:
            st.session_state.history.append({
                "question": query,
                "answer": "No matching internal information found."
            })


# ============================================================
# DISPLAY CHAT (NEWEST FIRST — NO HISTORY LABEL)
# ============================================================
for item in reversed(st.session_state.history):

    st.markdown("### ❓ Question")
    st.write(item["question"])

    st.markdown("### ✅ Answer")
    st.write(item["answer"])

    st.markdown("---")


# ============================================================
# REBUILD BUTTON
# ============================================================
if st.button("Rebuild Index"):
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    st.success("Index rebuilt successfully!")
