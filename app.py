import streamlit as st
import os

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
# CHAT HISTORY STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("<h4 style='margin-top: 30px;'>Select an option</h4>", unsafe_allow_html=True)
    mode = st.radio("", ["General", "Bayut", "Dubizzle"])


# ============================================================
# HEADER
# ============================================================
st.markdown("""
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
""", unsafe_allow_html=True)


# ============================================================
# DATA DIRECTORY HANDLING
# ============================================================
DATA_DIR = "/app/data"
LOCAL_FALLBACK = "./data"
if os.path.exists(LOCAL_FALLBACK):
    DATA_DIR = LOCAL_FALLBACK


# ============================================================
# EMBEDDINGS MODEL
# ============================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ============================================================
# VECTORSTORE PATH
# ============================================================
FAISS_DIR = os.path.join(DATA_DIR, "faiss_store")


# ============================================================
# BUILD VECTORSTORE
# ============================================================
def build_vectorstore():
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            try:
                with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                if not text.strip():
                    continue

                for chunk in splitter.split_text(text):
                    documents.append(Document(page_content=chunk, metadata={"source": file}))

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
# INITIALIZE VECTORSTORE
# ============================================================
vectorstore = load_vectorstore() or build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# ============================================================
# CLEAN ANSWER EXTRACTION
# ============================================================
def extract_clean_answer(raw_text, user_question):
    raw = raw_text.strip()
    user_q = user_question.lower()

    sections = raw.split("Q")
    best = ""

    for sec in sections:
        if user_q in sec.lower():
            best = sec
            break

    if not best:
        return raw

    if "–" in best:
        best = best.split("–", 1)[1].strip()

    return best


# ============================================================
# INPUT FIELD
# ============================================================
st.subheader(mode)
query = st.text_input("Ask your question:")


# ============================================================
# CENTERED BUTTONS (FIXED)
# ============================================================
colA, colB, colC, colD = st.columns([1, 1, 1, 1])

with colB:
    clear_clicked = st.button("🗑️ Clear Chat", use_container_width=True)

with colC:
    rebuild_clicked = st.button("🔄 Rebuild Index", use_container_width=True)


# ============================================================
# CLEAR CHAT (NOW WORKS 100%)
# ============================================================
if clear_clicked:
    st.session_state.history = []
    st.rerun()


# ============================================================
# REBUILD INDEX
# ============================================================
if rebuild_clicked:
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    st.success("Index rebuilt successfully!")
    st.rerun()


# ============================================================
# PROCESS QUERY
# ============================================================
if query:
    with st.spinner("Searching internal knowledge..."):
        docs = retriever.invoke(query)

        if docs:
            clean = extract_clean_answer(docs[0].page_content, query)
        else:
            clean = "No matching internal information found."

        st.session_state.history.append({"question": query, "answer": clean})


# ============================================================
# DISPLAY CHAT (NEWEST FIRST)
# ============================================================
for item in reversed(st.session_state.history):
    st.markdown("### ❓ Question")
    st.write(item["question"])

    st.markdown("### ✅ Answer")
    st.write(item["answer"])

    st.markdown("---")
