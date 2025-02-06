import os
import pdfplumber
import chainlit as cl
import requests
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize ChromaDB
vector_store = None

# Load the Hugging Face API token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the Mistral API endpoint and model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Helper function to make API requests to the Mistral model
def query_mistral_model(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "options": {"use_cache": False}}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for HTTP issues
    result = response.json()
    return result[0]['generated_text']

# Load the PDF file and create embeddings
def load_and_index_pdf(pdf_path: str):
    global vector_store
    
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for page in pdf.pages:
            texts.append(page.extract_text())
    
    # Create documents
    documents = [Document(page_content=text) for text in texts if text]
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_documents = text_splitter.split_documents(documents)
    
    # Create and store vectors
    vector_store = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

# Retrieve relevant documents based on a query
def retrieve_documents(query: str) -> str:
    if vector_store is None:
        raise ValueError("Vector store not initialized")
    
    # Get similar documents
    docs = vector_store.similarity_search(query, k=3)
    
    # Concatenate documents into one string for context
    return "\n\n".join([doc.page_content for doc in docs])

# Runs when the chat starts
@cl.on_chat_start
async def on_chat_start():
    try:
        # Send initial message
        await cl.Message(content="Loading and processing the PDF...").send()
        
        # Load and index a sample PDF
        pdf_path = "paper.pdf"  # Update this path
        load_and_index_pdf(pdf_path)
        
        # Send success message
        await cl.Message(content="✅ PDF processed successfully! Ready for questions.").send()
        
    except Exception as e:
        await cl.Message(content=f"Error during initialization: {str(e)}").send()
        raise

# Runs when a message is sent
@cl.on_message
async def on_message(message: cl.Message):
    query = message.content

    try:
        # Step 1: Retrieve relevant passages using embeddings
        relevant_passages = retrieve_documents(query)
        
        # Step 2: Formulate a prompt for the Mistral model
        prompt = (
            "Use the following information to answer the question, use only the "
            "provided document. Do not include any of the context or the prompt "
            "in your response.\n\n"
            f"Context:\n{relevant_passages}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        # Step 3: Generate the answer using the Mistral model
        result = query_mistral_model(prompt)

        # Process the result to extract only the answer
        answer = result.split("Answer:")[-1].strip()

        # Send the answer with sources
        await cl.Message(
            content=answer,
            elements=[
                cl.Text(name="Sources", content=relevant_passages, display="inline")
            ]
        ).send()
        
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

# Cleanup when the app stops
@cl.on_stop
def cleanup():
    try:
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
    except Exception:
        pass
