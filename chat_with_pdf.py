import os
import io
import time
from typing import List, Tuple, Dict, Any

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pypdf import PdfReader
from pathlib import Path

# -----------------------------
# Utility: Read files -> Documents
# -----------------------------
SUPPORTED_TYPES = {".txt", ".pdf"}

def read_txt(file) -> str:
    # file is a BytesIO-like object from Streamlit
    return file.read().decode("utf-8", errors="ignore")

def read_pdf(file) -> str:
    reader = PdfReader(file)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def load_documents(uploaded_files: List[Any]) -> List[Document]:
    docs: List[Document] = []
    for f in uploaded_files:
        name = f.name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_TYPES:
            st.warning(f"Skipping unsupported file type: {name}")
            continue

        if suffix == ".txt":
            raw = read_txt(f)
        else:
            raw = read_pdf(f)

        if not raw.strip():
            st.warning(f"No extractable text in {name}.")
            continue

        metadata = {"source": name}
        docs.append(Document(page_content=raw, metadata=metadata))
    return docs


# -----------------------------
# Vector DB helpers
# -----------------------------
def build_vectorstore(docs: List[Document], persist_dir: str) -> Chroma:
    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Embeddings / DB
    embeddings = OpenAIEmbeddings(model="openai.text-embedding-3-large")  # uses OPENAI_API_KEY
    vector_db = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=persist_dir
    )
    vector_db.persist()
    return vector_db


# -----------------------------
# Retrieval + Generation
# -----------------------------
RAG_SYSTEM_PROMPT = """You are a helpful assistant answering questions using the provided context.
Follow the rules:
- Use only the context to answer; if the answer is not in the context, say you don't know.
- Cite relevant sources by filename and (if available) page hints.
- Keep answers concise but complete.
"""

def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        title = f"{src}" + (f" (page {page})" if page is not None else "")
        blocks.append(f"[{i}] Source: {title}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def answer_query(llm: ChatOpenAI, retriever, query: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[Document]]:
     # LC >= 0.2 retrievers are Runnables -> use .invoke()
    try:
        retrieved = retriever.invoke(query)  # List[Document]
    except AttributeError:
        # Back-compat with older LC retrievers
        retrieved = retriever.get_relevant_documents(query)

    ctx = format_context(retrieved)

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Chat history: {chat_history}\n\nQuestion: {query}\n\nContext:\n{ctx}"}
    ]
    resp = llm.invoke(messages)
    content = resp.content if hasattr(resp, "content") else str(resp)
    return content, retrieved


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Chat (LangChain + Chroma + Streamlit)", page_icon="💬", layout="wide")

st.title("💬 Retrieval-Augmented Chat over your Documents")
st.caption("LangChain • ChromaDB • Streamlit")

with st.sidebar:
    st.header("1) Upload documents")
    uploaded = st.file_uploader(
        "Upload .txt and/or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="You can add multiple files. We'll chunk and index them locally for retrieval."
    )

    st.header("2) Build index")
    persist_dir = st.text_input("Chroma persist directory", value="./.chroma")
    build_clicked = st.button("Build / Rebuild Index")

    st.markdown("---")
    st.header("Settings")
    model_name = st.text_input("OpenAI Chat Model", value="openai.gpt-4o")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.caption("Environment:\n- Set OPENAI_API_KEY in your Codespace or local shell.")

# Session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts {role, content}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (user, assistant)
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Build index
if build_clicked:
    if not uploaded:
        st.warning("Please upload at least one file first.")
    else:
        with st.spinner("Reading and indexing documents..."):
            docs = load_documents(uploaded)
            if not docs:
                st.error("No valid documents to index.")
            else:
                st.session_state.vector_db = build_vectorstore(docs, persist_dir=persist_dir)
                st.session_state.retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})
        if st.session_state.retriever is not None:
            st.success("Index ready! You can start chatting below.")

# Chat area
st.subheader("Chat")
clear = st.button("Clear chat")
if clear:
    st.session_state.messages = []
    st.session_state.chat_history = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask a question about your documents...")
if user_msg:
    if st.session_state.retriever is None:
        st.error("Please upload files and click 'Build / Rebuild Index' first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = ChatOpenAI(model=model_name, temperature=temperature)
            answer, sources = answer_query(llm, st.session_state.retriever, user_msg, st.session_state.chat_history)

            st.markdown(answer)

            # Source attributions
            if sources:
                with st.expander("Sources used"):
                    for d in sources:
                        src = d.metadata.get("source", "unknown")
                        page = d.metadata.get("page", None)
                        st.write(f"- **{src}**" + (f", page {page}" if page is not None else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append((user_msg, answer))