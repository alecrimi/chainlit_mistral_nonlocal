import os
import requests
import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Chainlit configuration to hide the readme button
cl.config.show_readme = False

# Initialize global variables
vector_store = None
embeddings = None

# Load the Hugging Face API token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the Mistral API endpoint
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

def query_mistral_model(prompt: str) -> str:
    """Make API requests to the Mistral model"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "options": {
            "use_cache": False,
            "max_length": 1000,
            "temperature": 0.7
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result[0]['generated_text']

def load_vector_store(persist_directory: str = "./chroma_db"):
    """Load the existing Chroma vector store"""
    global vector_store, embeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store

def retrieve_documents(query: str) -> str:
    """Retrieve relevant documents based on a query"""
    if vector_store is None:
        raise ValueError("Previous knowledge initialized")
    
    docs = vector_store.similarity_search(query, k=5)
    return "\n\n".join([doc.page_content for doc in docs])

def clean_response(response: str) -> str:
    """Clean the response to remove any source text or context"""
    # List of potential markers that might appear in the response
    markers = [
        "Context:", "Source:", "Reference:", "Document:", 
        "Here's the relevant information:", "Based on the provided context:",
        "According to the document:", "From the text:"
    ]
    
    cleaned = response
    
    # Remove everything after any of the markers
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]
    
    # Remove everything between {{ and }}
    while "{{" in cleaned and "}}" in cleaned:
        start = cleaned.find("{{")
        end = cleaned.find("}}") + 2
        cleaned = cleaned[:start] + cleaned[end:]
    
    return cleaned.strip()

@cl.on_chat_start
async def on_chat_start():
    try:
        await cl.Message(content="Loading the vector database...").send()
        load_vector_store()
        await cl.Message(content="✅ Vector store loaded successfully! Ready for questions.").send()
    except Exception as e:
        await cl.Message(content=f"Error during initialization: {str(e)}").send()
        raise

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content

    try:
        # Retrieve relevant passages
        relevant_passages = retrieve_documents(query)
        
        # Modified prompt to explicitly instruct not to include sources
        prompt = (
            "You are a highly knowledgeable expert. Using the provided context, create a "
            "detailed response that answers the question comprehensively. Do not quote or "
            "reference the source material directly. Do not mention that you're using any "
            "context or sources. Simply provide the information as if it's your own "
            "knowledge. Structure your response in clear paragraphs.\n\n"
            f"Context for your reference (do not mention or quote this):\n{relevant_passages}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        # Generate answer
        result = query_mistral_model(prompt)
        
        # Clean and process the response
        answer = result.split("Answer:")[-1].strip()
        cleaned_answer = clean_response(answer)
        
        # If the answer is too short, try to get more details
        if len(cleaned_answer.split()) < 50:
            followup_prompt = (
                f"Please provide more details about this topic, without mentioning any sources: {cleaned_answer}\n"
                "Expanded answer:"
            )
            result = query_mistral_model(followup_prompt)
            followup_answer = result.split("Expanded answer:")[-1].strip()
            cleaned_answer = clean_response(followup_answer)

        # Send only the cleaned response
        await cl.Message(content=cleaned_answer).send()
        
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

if __name__ == "__main__":
    cl.run()