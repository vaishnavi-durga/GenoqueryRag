from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the 'backend' folder to the Python path
# This allows us to import from faiss_store.py and rag.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.faiss_store import search_faiss_index
from backend.rag import generate_answer

# --- FastAPI App Setup ---
app = FastAPI(
    title="MedQuickConsult API",
    description="API for the RAG medical assistant."
)

# --- IMPORTANT: CORS ---
# This allows your frontend (on a different URL) to talk to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g., localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET)
    allow_headers=["*"],
)

# --- Pydantic Models (The "API Contract") ---
class QueryRequest(BaseModel):
    query_text: str
    mode: str = "Doctor" # Default to "Doctor"

class QueryResponse(BaseModel):
    answer: str
    sources: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "MedQuickConsult API is running."}


@app.post("/api/v1/generate_report", response_model=QueryResponse)
async def generate_report(request: QueryRequest):
    """
    The main RAG endpoint.
    Receives a query and mode, returns an answer and sources.
    """
    try:
        print(f"Received query: {request.query_text} (Mode: {request.mode})")
        
        # 1. Search FAISS
        retrieved_chunks = search_faiss_index(request.query_text, k=5)
        
        if not retrieved_chunks:
            return {"answer": "I could not find any relevant information in the provided documents to answer your question.", "sources": ""}
            
        # 2. Generate Answer
        result = generate_answer(request.query_text, retrieved_chunks, request.mode)
        
        return {"answer": result["answer"], "sources": result["sources"]}
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# To run this server:
# 1. Make sure your (venv) is active
# 2. In your terminal, run:
#    uvicorn api.app:app --reload