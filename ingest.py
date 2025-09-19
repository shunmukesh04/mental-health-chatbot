import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_vectorstore

DATA_DIR = Path("data")
SUPPORTED_EXTS = {".pdf", ".txt", ".md"}


def load_documents() -> List:
    documents = []
    if not DATA_DIR.exists():
        return documents
    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
        documents.extend(docs)
    return documents


def main() -> None:
    print("Loading documents from ./data ...")
    docs = load_documents()
    if not docs:
        print("No documents found. Place .pdf, .txt, or .md files in ./data")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks. Writing to Chroma...")

    vectordb = get_vectorstore()
    vectordb.add_documents(chunks)
    vectordb.persist()
    print("Ingestion complete. Vector store persisted.")


if __name__ == "__main__":
    main()
