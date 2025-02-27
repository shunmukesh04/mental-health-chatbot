import streamlit as st
import os
import tempfile
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores.chroma import Chroma, chroma_client_settings


# Function to Initialize the Language Model
def initialize_llm():
    groq_api_key = st.secrets[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]  # Use secrets instead of hardcoding
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama-3-70b-versatile"
    )
    return llm


# Function to Create Vector Database
def create_vector_db(uploaded_files):
    documents = []
    
    for uploaded_file in uploaded_files:
        # Save file to temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load PDF content
        loader = PyPDFLoader(temp_file_path)
        documents.extend(loader.load())

        # Remove temporary file
        os.remove(temp_file_path)

    if not documents:
        raise ValueError("No text extracted from PDFs. Check file format.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Load Embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma vector database
    vector_db = Chroma.from_documents(
        texts, embeddings, persist_directory="./chroma_db",
        client_settings=chroma_client_settings  # Correct import for settings
    )

    vector_db.persist()
    return vector_db


# Function to Setup the Question-Answering Chain
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


# Streamlit UI
st.title("Mental Health Chatbot")

# Session state for QA chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if st.button("Process PDFs"):
    if uploaded_files:
        llm = initialize_llm()
        vector_db = create_vector_db(uploaded_files)
        st.session_state.qa_chain = setup_qa_chain(vector_db, llm)
        st.success("PDFs processed successfully!")
    else:
        st.error("Please upload at least one PDF file.")

# Query input
query = st.text_input("Ask your question:")
if st.button("Get Answer"):
    if st.session_state.qa_chain:
        response = st.session_state.qa_chain.run(query)
        st.text_area("Chatbot Response:", value=response, height=150)
    else:
        st.error("Please process PDFs first.")
