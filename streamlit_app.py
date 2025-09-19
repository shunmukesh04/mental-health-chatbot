import os
from pathlib import Path
from typing import List

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    CHROMA_PERSIST_DIR,
    get_vectorstore,
)
from rag_chain import build_rag_chain

st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="wide")

DATA_DIR = Path("data")
SUPPORTED_EXTS = {".pdf", ".txt", ".md"}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)


def add_files_to_vectorstore(files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    vectordb = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    new_chunks = []

    for f in files:
        suffix = Path(f.name).suffix.lower()
        tmp_path = Path(".tmp") / f.name
        tmp_path.parent.mkdir(exist_ok=True)
        with open(tmp_path, "wb") as out:
            out.write(f.getbuffer())
        if suffix == ".pdf":
            loader = PyPDFLoader(str(tmp_path))
            docs = loader.load()
        else:
            loader = TextLoader(str(tmp_path), encoding="utf-8")
            docs = loader.load()
        new_chunks.extend(splitter.split_documents(docs))
        tmp_path.unlink(missing_ok=True)

    if new_chunks:
        vectordb.add_documents(new_chunks)
        vectordb.persist()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Support")
        st.markdown(
            "If you're struggling or in crisis, please reach out for immediate help."
        )
        st.markdown(
            "- **Emergency**: Call your local emergency number.\n"
            "- **988 Suicide & Crisis Lifeline (US)**: Call or text 988, or chat at 988lifeline.org"
        )
        st.divider()
        st.subheader("Daily Quote")
        st.markdown("“No feeling is final.” — Rainer Maria Rilke")


def main():
    ensure_dirs()

    st.title("🧠 Mental Health Chatbot")
    st.caption("Supportive, empathetic assistant with retrieval-augmented responses. Not medical advice.")

    render_sidebar()

    api_key = GROQ_API_KEY
    model_name = GROQ_MODEL

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("How are you feeling today?"):
        if not api_key:
            st.error("Please set GROQ_API_KEY in your .env file")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chain = build_rag_chain(groq_api_key=api_key, groq_model=model_name)
            placeholder = st.empty()
            full_response = ""
            for chunk in chain.stream(prompt):
                full_response += chunk
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.caption(
        "If you're in crisis, please contact local emergency services or a crisis hotline immediately."
    )


if __name__ == "__main__":
    main()
