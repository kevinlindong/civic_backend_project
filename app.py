# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Import the documents from data.py
from data import DOCUMENTS

# Initialize FastAPI app
app = FastAPI()

# Initialize embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define document model
class Document(BaseModel):
    id: int
    text: str

# Simple vector store
class SimpleVectorStore:
    def __init__(self):
        # Dimension of embeddings from the model
        self.dimension = 384
        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(self.dimension)
        # Store document texts
        self.documents = {}
        # Map FAISS index positions to document IDs
        self.index_to_id = {}
    
    def add_document(self, doc: Document):
        # Convert text to embedding vector
        embedding = model.encode([doc.text])[0].astype(np.float32)
        
        # Current index position
        idx = self.index.ntotal
        
        # Add embedding to FAISS index
        self.index.add(np.array([embedding]).reshape(1, self.dimension))
        
        # Store the document and its mapping
        self.documents[doc.id] = doc.text
        self.index_to_id[idx] = doc.id
    
    def query(self, text: str, top_k: int = 1):
        # Convert query to embedding
        query_vector = model.encode([text])[0].astype(np.float32)
        
        # Search for similar vectors
        distances, indices = self.index.search(np.array([query_vector]), top_k)
        
        # Prepare results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:  # Valid index
                doc_id = self.index_to_id[idx]
                results.append({
                    "id": doc_id,
                    "text": self.documents[doc_id],
                    "score": float(1 / (1 + distances[0][i]))  # Convert distance to similarity score
                })
        
        return results

# Create vector store instance
vector_store = SimpleVectorStore()

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Thank you for this opportunity and I hope you enjoy this assignment!"}

@app.post("/ingest")
def ingest_document(doc: Document):
    try:
        vector_store.add_document(doc)
        return {"status": "success", "message": f"Document {doc.id} added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query")
def query_documents(text: str, top_k: int = 1):
    try:
        results = vector_store.query(text, top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load initial data
@app.on_event("startup")
def load_initial_data():
    for doc_data in DOCUMENTS:
        doc = Document(**doc_data)
        vector_store.add_document(doc)
    
    print("Initial data loaded")
