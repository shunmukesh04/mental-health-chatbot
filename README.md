# 🧠 Mental Health Chatbot (LangChain + Chroma + Groq + Streamlit)

A supportive, empathetic RAG chatbot using LangChain, ChromaDB, Groq LLMs, and Streamlit.
![WhatsApp Image 2025-09-19 at 11 03 38_1384ce63](https://github.com/user-attachments/assets/f8da64ae-c73a-444f-a2e9-e2e2f0a51d5c)
![WhatsApp Image 2025-09-19 at 10 51 46_3d4142f4](https://github.com/user-attachments/assets/737dd377-7859-4a7d-9ef0-ce7f5437fd1f)


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

