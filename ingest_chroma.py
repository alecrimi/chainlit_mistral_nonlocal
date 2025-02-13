import os
import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_pdf(pdf_path: str, persist_directory: str = "./chroma_db"):
    """
    Ingest a PDF file and create a Chroma vector database.
    
    Args:
        pdf_path: Path to the PDF file
        persist_directory: Directory to store the Chroma database
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for page in pdf.pages:
            texts.append(page.extract_text())
    
    # Create documents
    documents = [Document(page_content=text) for text in texts if text]
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )
    split_documents = text_splitter.split_documents(documents)
    
    # Create and store vectors
    vector_store = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✅ PDF processed and stored in {persist_directory}")
    return vector_store

if __name__ == "__main__":
    pdf_path = "paper.pdf"  # Update this path
    ingest_pdf(pdf_path)