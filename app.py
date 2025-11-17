# =========================================================
# Bayut & Dubizzle AI Content Assistant — Internal RAG Only
# =========================================================

import os
import shutil
import difflib
import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from uuid import uuid4


# ---------------- Page config ----------------
st.set_page_config(
    page_title="Bayut & Dubizzle AI Content Assistant",
    layout="wide"
)


# ---------------- CSS ----------------
st.markdown("""
<style>

[data-testid="stVerticalBlock"] > div {
    background: transparent !important;
    box-shadow:none!important;
    padding:0!important;
    margin:0!important;
}

main .block-container {
    padding-top:0rem !important;
}

/* Chat bubbles */
.bubble {
    padding:12px 16px;
    border-radius:14px;
    margin:6px 0;
    max-width:85%;
    line-height:1.6;
    font-size:15px;
}
.bubble.user { background:#f2f2f2; margin-left:auto; }
.bubble.ai { background:#ffffff; margin-right:auto; }

.evidence {
    background:#fafafa;
    border:1px solid #e5e7eb;
    border-radius:12px;
    padding:10px;
    margin:10px 0;
    font-size:13px;
    white-space:pre-wrap;
}

.sidebar-label {
    font-size: 16px;
    padding: 6px 0;
    font-weight: 600;
}

.sidebar-general { color:#000000; }
.sidebar-bayut { color:#008060; }
.sidebar-dubizzle { color:#D92C27; }

/* ---------------- BUTTON FIX ---------------- */
.center-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)



# ---------------- Header ----------------
st.markdown("""
<div style='text-align:center; font-size:38px; font-weight:900; margin-bottom:0;'>
    <span style='color:#008060;'>Bayut</span>
    <span style='color:#000000;'> & </span>
    <span style='color:#D92C27;'>Dubizzle</span>
    <span style='color:#000000;'> AI Content Assistant</span>
</div>

<p style='text-align:center; color:#555; margin-top:-6px; font-size:15px;'>
Fast internal knowledge search powered by internal content.
</p>
""", unsafe_allow_html=True)



# ---------------- Sidebar ----------------
st.sidebar.markdown("#### Select an option")

tool = st.sidebar.radio("", ["General", "Bayut", "Dubizzle"])

if tool == "General":
    st.sidebar.markdown("<p class='sidebar-label sidebar-general'>General</p>", unsafe_allow_html=True)
elif tool == "Bayut":
    st.sidebar.markdown("<p class='sidebar-label sidebar-bayut'>Bayut</p>", unsafe_allow_html=True)
elif tool == "Dubizzle":
    st.sidebar.markdown("<p class='sidebar-label sidebar-dubizzle'>Dubizzle</p>", unsafe_allow_html=True)



# ---------------- Vectorstore Paths ----------------
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")


def find_best_matching_file(query, folder=DATA_DIR):
    if not os.path.isdir(folder): return None
    files = [f for f in os.listdir(folder) if f != "faiss_store"]
    if not files: return None
    match = difflib.get_close_matches(query.lower(), [f.lower() for f in files], n=1, cutoff=0.3)
    for f in files:
        if f.lower() == match[0]: return os.path.join(folder, f)
    return None


# ---------------- LangChain RAG ----------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class DummyLLM:
    def invoke(self, text):
        marker = "ANSWER (STRICTLY FROM CONTEXT):"
        if isinstance(text, dict): text = str(text)
        return text.split(marker, 1)[-1].strip() if marker in text else text


def get_local_llm():
    if os.getenv("USE_DUMMY_LLM", "0") == "1": 
        return DummyLLM()
    from langchain_groq import ChatGroq
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant", temperature=0)


def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf": return PyPDFLoader(path).load()
        if ext == ".docx": return Docx2txtLoader(path).load()
        if ext in [".txt",".md"]: return TextLoader(path,autodetect_encoding=True).load()
        if ext == ".csv":
            for enc in ["utf-8", "utf-8-sig", "cp1256", "windows-1256", None]:
                try: return CSVLoader(path, encoding=enc).load()
                except: continue
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [Document(page_content=df.to_string(index=False), metadata={"source": path})]
        return []
    except:
        return []


def load_default_docs():
    docs=[]
    if not os.path.isdir(DATA_DIR): return docs
    for f in os.listdir(DATA_DIR):
        if f=="faiss_store": continue
        full=os.path.join(DATA_DIR,f)
        if os.path.isfile(full): docs.extend(load_document(full))
    return docs


def faiss_exists(): return os.path.isdir(INDEX_DIR)


@st.cache_resource
def build_vectorstore(docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=50)
    chunks=splitter.split_documents(docs)
    return FAISS.from_documents(chunks,get_embeddings())


def reset_app():
    st.session_state.pop("rag_history",None)
    st.rerun()



# ---------------- Load Index ----------------
if faiss_exists():
    vectorstore = FAISS.load_local(INDEX_DIR,get_embeddings(),allow_dangerous_deserialization=True)
else:
    docs = load_default_docs()
    if not docs:
        st.error("No documents in /data folder.")
        st.stop()
    vectorstore = build_vectorstore(docs)
    vectorstore.save_local(INDEX_DIR)



# ---------------- Chat History ----------------
if "rag_history" not in st.session_state:
    st.session_state["rag_history"] = []


# ---------------- UI ----------------
st.write(f"### {tool}")

for q,a in st.session_state["rag_history"]:
    st.markdown(f"<div class='bubble user'>{q}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{a}</div>",unsafe_allow_html=True)

query = st.text_input("Ask your question:")


# ----------- BUTTONS (Guaranteed Horizontal + Centered) -----------

# Create 3 equal columns in the center
col_spacer_left, col1, col2, col3, col_spacer_right = st.columns([2,1,1,1,2])

with col1:
    b1 = st.button("Rebuild Index")

with col2:
    b2 = st.button("Clear Chat")

with col3:
    b3 = st.button("Reload")


# ---------------- Run Query ----------------
if query:

    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    extract = RunnableLambda(lambda x: x["question"])
    docsearch = extract | retriever

    def join_docs(docs):
        return "\n\n".join(d.page_content[:1800] for d in docs)

    context = docsearch | RunnableLambda(join_docs)

    prompt = PromptTemplate.from_template("""
You are an internal assistant. Use ONLY the provided context.
If the answer is missing, reply: "This information is not available in internal content."

==========================
CONTEXT:
{context}
==========================
QUESTION:
{question}
==========================
ANSWER (STRICTLY FROM CONTEXT):
""")

    chain = (
        {"context": context, "question": RunnablePassthrough()}
        | prompt
        | get_local_llm()
        | StrOutputParser()
    )

    answer = chain.invoke({"question": query})

    st.session_state["rag_history"].append((query, answer))

    st.markdown(f"<div class='bubble user'>{query}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='bubble ai'>{answer}</div>",unsafe_allow_html=True)
