import os
from dotenv import load_dotenv

# Try to load .env for local runs
load_dotenv()

# Prefer Streamlit secrets when available (Streamlit Cloud), fallback to env vars
try:
	import streamlit as st  # type: ignore
	_secrets = st.secrets  # type: ignore[attr-defined]
except Exception:
	_secrets = {}


def _get_secret(name: str, default: str | None = None) -> str | None:
	try:
		if _secrets and name in _secrets:
			return str(_secrets.get(name))
	except Exception:
		pass
	return os.getenv(name, default)


GROQ_API_KEY: str | None = _get_secret("GROQ_API_KEY")
CHROMA_PERSIST_DIR: str = _get_secret("CHROMA_PERSIST_DIR", "./chroma_db") or "./chroma_db"
EMBEDDINGS_MODEL: str = _get_secret("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL: str = _get_secret("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile"
VECTORSTORE: str = (_get_secret("VECTORSTORE", "chroma") or "chroma").lower()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def get_embeddings() -> HuggingFaceEmbeddings:
	return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


def get_vectorstore(persist_directory: str | None = None):
	if VECTORSTORE == "none":
		return None
	# default to chroma
	return Chroma(
		persist_directory=persist_directory or CHROMA_PERSIST_DIR,
		embedding_function=get_embeddings(),
	)
