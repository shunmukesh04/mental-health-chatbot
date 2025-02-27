import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
import tempfile

def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",  # Replace with your actual API key
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vector_db.persist()
    return vector_db


def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

st.title("Mental Health Chatbot")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
if st.button("Process PDFs"):
    if uploaded_files:
        llm = initialize_llm()
        vector_db = create_vector_db(uploaded_files)
        st.session_state.qa_chain = setup_qa_chain(vector_db, llm)
        st.success("PDFs processed successfully!")
    else:
        st.error("Please upload at least one PDF file.")

query = st.text_input("Ask your question:")
if st.button("Get Answer"):
    if st.session_state.qa_chain:
        response = st.session_state.qa_chain.run(query)
        st.text_area("Chatbot Response:", value=response, height=150)
    else:
        st.error("Please process PDFs first.")
