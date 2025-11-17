# =========================================================
# app.py — Bayut & Dubizzle AI Content Assistant (Internal RAG Only)
# =========================================================

import os
import shutil
import difflib
import pandas as pd
import streamlit as st
from uuid import uuid4
from langchain_core.documents import Document

# ---------------- Page config ----------------
st.set_page_config(page_title="Bayut & Dubizzle AI Content Assistant", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
[data-testid="stVerticalBlock"] > div { background: transparent !important; box-shadow:none!important; padding:0!important;margin:0!important;}
main .block-container { padding-top:0rem !important;}
.bubble {padding:12px 16px;border-radius:14px;margin:6px 0;max-width:85%;line-height:1.6;font-size:15px;}
.bubble.user {background:#f2f2f2;margin-left:auto;}
.bubble.ai {background:#ffffff;margin-right:auto;}
.evidence {background:#fafafa;border:1px solid #e5e7eb;border-radius:12px;padding:10px;margin:10px 0;font-size:13px;white-space:pre-wrap;}
.title-red {font-size:34px;font-weight:900;color:#D92C27;text-align:center;}
.title-main {font-size:28px;font-weight:700;color:#111;text-align:center;margin-top:-10px;}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<div class='title-red'>Dubizzle Group AI Lab</div>", unsafe_allow_html=True)
st.markdown("<div class='title-main'>Bayut & Dubizzle AI Content Assistant</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Fast internal knowledge search powered by internal content.</p>", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab_general, tab_bayut, tab_dubizzle = st.tabs([
    "🌍 General Assistant",
    "🏡 Bayut Assistant",
    "🚗 Dubizzle Assistant"
])

# ---------------- File + Vectorstore Paths ----------------
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

def find_best_matching_file(query, folder=DATA_DIR):
    if not os.path.isdir(folder):
        return None
    files = [f for f in os.listdir(folder) if f != "faiss_store"]
    if not files:
        return None
    match = difflib.get_close_matches(query.lower(), [f.lower() for f in files], n=1, cutoff=0.3)
    if not match:
        return None
    for f in files:
        if f.lower() == match[0]:
            return os.path.join(folder, f)
    return None

# ---------------- LangChain / RAG Components ----------------
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
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant", temperature=0.0, top_p=0.0)


def load_document(path: str):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":  return PyPDFLoader(path).load()
        if ext == ".docx": return Docx2txtLoader(path).load()
        if ext in [".txt", ".md"]: return TextLoader(path, autodetect_encoding=True).load()
        if ext == ".csv":
            for enc in ["utf-8","utf-8-sig","cp1256","windows-1256",None]:
                try: return CSVLoader(path, encoding=enc).load()
                except: continue
            return []
        if ext == ".xlsx":
            df = pd.read_excel(path)
            return [Document(page_content=df.to_string(index=False), metadata={"source": path})]
        return []
    except:
        return []


def load_default_docs():
    docs = []
    if not os.path.isdir(DATA_DIR): return docs
    for f in os.listdir(DATA_DIR):
        if f == "faiss_store": continue
        full = os.path.join(DATA_DIR, f)
        if os.path.isfile(full): docs.extend(load_document(full))
    return docs


def faiss_exists(): return os.path.isdir(INDEX_DIR)


@st.cache_resource
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


def save_faiss(store): store.save_local(INDEX_DIR)
def load_faiss(): return FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)


def _reload():
    st.rerun()


def _reset():
    st.session_state.pop("rag_history", None)
    _reload()

# ---------------- Load / Create Vectorstore ----------------
if faiss_exists():
    vectorstore = load_faiss()
    st.success("✅ Index loaded")
else:
    docs = load_default_docs()
    if not docs:
        st.error("❌ No documents found in /data folder.")
        st.stop()
    vectorstore = build_vectorstore(docs)
    save_faiss(vectorstore)
    st.success("✅ Index created")


# ---------------- Chat History ----------------
if "rag_history" not in st.session_state:
    st.session_state["rag_history"] = []


def chat_ui(tab_label, key_name):
    with tab_label:
        for q, a in st.session_state["rag_history"]:
            st.markdown(f"<div class='bubble user'>{q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bubble ai'>{a}</div>", unsafe_allow_html=True)

        query = st.text_input("Ask your question:", key=key_name)

        st.write("")
        left, mid, right = st.columns([1,2,1])
        with mid:
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                rebuild = st.button("🔄 Rebuild Index", key=f"rebuild_{key_name}")
            with c2:
                st.button("🧹 Clear Chat", key=f"clear_{key_name}", on_click=_reset, use_container_width=True)
            with c3:
                st.button("🔁 Reload", key=f"reload_{key_name}", on_click=_reload, use_container_width=True)

        if rebuild:
            shutil.rmtree(INDEX_DIR)
            st.cache_resource.clear()
            st.success("Index removed — reload page.")
            st.stop()

        if query:
            if any(x in query.lower() for x in ["download","file","send","share","get"]):
                match = find_best_matching_file(query)
                if match:
                    st.success(f"File found: {os.path.basename(match)}")
                    with open(match,"rb") as f:
                        st.download_button("⬇️ Download", f, file_name=os.path.basename(match))
                else:
                    st.error("No matching file found.")
                st.stop()

            with st.spinner("Thinking…"):
                retriever = vectorstore.as_retriever(search_kwargs={"k":3})
                extract = RunnableLambda(lambda x: x["question"])
                retrieve_docs = extract | retriever

                def join_docs(docs):
                    return "\n\n".join(d.page_content[:1800] for d in docs) if docs else ""

                context = retrieve_docs | RunnableLambda(join_docs)

                prompt = PromptTemplate.from_template("""
You are an internal assistant. Only answer using the CONTEXT.
If the answer is not found, respond: "This information is not available in the internal content."

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
                    {"context": context, "question":RunnablePassthrough()}
                    | prompt
                    | get_local_llm()
                    | StrOutputParser()
                )

                answer = chain.invoke({"question": query})
                st.session_state["rag_history"].append((query, answer))

                st.markdown(f"<div class='bubble user'>{query}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='bubble ai'>{answer}</div>", unsafe_allow_html=True)

                hits = vectorstore.similarity_search_with_score(query, k=3)
                if hits:
                    st.markdown("### 📎 Evidence")
                    for i, (doc, score) in enumerate(hits, 1):
                        snippet = doc.page_content[:500]
                        st.markdown(f"<div class='evidence'><b>{i}.</b> (score={score:.3f})\n\n{snippet}…</div>", unsafe_allow_html=True)


chat_ui(tab_general, "general")
chat_ui(tab_bayut, "bayut")
chat_ui(tab_dubizzle, "dubizzle")
