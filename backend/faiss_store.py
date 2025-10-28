import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle

# --- Constants ---
FAISS_INDEX_PATH = "backend/faiss_index.idx"
METADATA_PATH = "backend/metadata.pkl"
# Using a fast model for the hackathon
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 

# --- Load Model (this will run once when the file is imported) ---
print("Loading embedding model (this may take a moment)...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded.")


def build_faiss_index(chunks: list[dict]):
    """
    Takes text chunks, creates embeddings, and builds a FAISS index.
    """
    if not chunks:
        print("Error: No chunks provided to build index.")
        return

    print(f"Building FAISS index for {len(chunks)} chunks...")
    
    # 1. Get all text for embedding
    texts = [chunk['text'] for chunk in chunks]
    
    # 2. Create embeddings
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # FAISS requires float32
    embeddings = embeddings.astype('float32') 
    
    # 3. Build FAISS index
    d = embeddings.shape[1] # Dimension of embeddings
    # Simple L2 (Euclidean) distance index
    index = faiss.IndexFlatL2(d) 
    index.add(embeddings)
    
    # 4. Save the index and metadata
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Store metadata (text, source, page) separately
    # We use pickle to easily save/load the list of metadata dicts
    metadata_list = [chunk['metadata'] for chunk in chunks]
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata_list, f)
        
    print(f"Successfully built and saved FAISS index and metadata.")


def search_faiss_index(query_text: str, k: int = 5) -> list[dict]:
    """
    Searches the FAISS index for the top k most similar chunks.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Error: FAISS index not found. Please build it first.")
        return []
        
    try:
        # 1. Load index and metadata
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
            
        # 2. Create query embedding
        query_embedding = model.encode([query_text], convert_to_numpy=True).astype('float32')
        
        # 3. Search the index
        # D = distances, I = indices of the chunks
        D, I = index.search(query_embedding, k)
        
        # 4. Format results
        results = []
        for i in I[0]: # I[0] contains the list of indices
            # Get the metadata for the matching chunk
            results.append(metadata[i]) 
            
        print(f"Found {len(results)} relevant chunks.")
        return results
        
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []