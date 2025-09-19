# 🧠 Mental Health Chatbot (LangChain + Chroma + Groq + Streamlit)

A supportive, empathetic RAG chatbot using LangChain, ChromaDB, Groq LLMs, and Streamlit.

## Prerequisites
- Python 3.10+
- A Groq API key (get one at `https://console.groq.com`)

## Setup
```bash
# 1) Create and activate a virtualenv (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure environment
copy .env.example .env
# Edit .env and set GROQ_API_KEY

# 4) (Optional) Ingest any local docs placed in ./data
python ingest.py

# 5) Run the Streamlit app
streamlit run streamlit_app.py
```

## Usage
- Use the sidebar to set your Groq API key and model.
- Upload .pdf/.txt/.md documents to expand the knowledge base.
- Chat in the main panel. Responses are RAG-augmented and streamed.

## Notes
- This app is not a substitute for professional mental health care.
- In emergencies, contact local emergency services or a crisis hotline.

