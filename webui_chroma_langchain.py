import os
import pdfplumber
import requests

from fastapi import APIRouter, HTTPException
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Create a FastAPI router that you can later include into Open‑webUI’s app.
router = APIRouter()

# -----------------------------------------------------------------------------
# Helper function: Query the Mistral model via Hugging Face's Inference API
# -----------------------------------------------------------------------------
def query_mistral_model(prompt: str) -> str:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set.")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    payload = {"inputs": prompt, "options": {"use_cache": False}}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    # Assumes the response JSON is a list with a dict containing 'generated_text'
    return result[0]['generated_text']

# -----------------------------------------------------------------------------
# Helper function: Load and split the PDF into document chunks
# -----------------------------------------------------------------------------
def load_pdf_documents(pdf_path: str):
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text))
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {str(e)}")
    if not documents:
        raise ValueError("No text found in PDF.")
    return documents

# -----------------------------------------------------------------------------
# Build the vector store on startup
# -----------------------------------------------------------------------------
# Adjust the PDF file path as needed
PDF_PATH = "paper.pdf"
raw_documents = load_pdf_documents(PDF_PATH)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Initialize embeddings model and build the Chroma vector store.
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents,
    embedding=embeddings,
    collection_name="pdf_docs",
    persist_directory="./chroma_db"
)

# -----------------------------------------------------------------------------
# Helper function: Retrieve relevant passages from the vector store
# -----------------------------------------------------------------------------
def retrieve_documents(query: str, k: int = 3) -> str:
    retrieved_docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

# -----------------------------------------------------------------------------
# FastAPI endpoint: RAG query
# -----------------------------------------------------------------------------
@router.get("/rag")
def rag_query(query: str):
    try:
        # Retrieve context from the vector store
        context = retrieve_documents(query)
        # Build the prompt for Mistral. The prompt instructs the model to answer based only on the provided context.
        prompt = (
            "Use the following information to answer the question. "
            "Do not include the context or prompt text in your answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        # Query the Mistral model
        result = query_mistral_model(prompt)
        # Optionally, extract only the answer portion if extra tokens are returned
        answer = result.split("Answer:")[-1].strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
