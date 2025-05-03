# imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from data import DOCUMENTS


app = FastAPI()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#define document model
class Document(BaseModel):
    id: int
    text: str

# vector store
class VectorStore:
    def __init__(self):
        # load the model and create the index
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = {}
        self.index_to_id = {}
    
    def add_document(self, doc: Document):
        # convert text to embedding vector
        embedding = model.encode([doc.text])[0].astype(np.float32)
        
        # current index position
        idx = self.index.ntotal
        # add embedding to FAISS index
        self.index.add(np.array([embedding]).reshape(1, self.dimension))
        
        # store the document and its mapping
        self.documents[doc.id] = doc.text
        self.index_to_id[idx] = doc.id
    
    def query(self, text: str, top_k: int = 1):
        # convert query to embedding
        query_vector = model.encode([text])[0].astype(np.float32)
        
        # search for similar vectors
        distances, indices = self.index.search(np.array([query_vector]), top_k)
        
        # prep results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                doc_id = self.index_to_id[idx]
                results.append({
                    "id": doc_id,
                    "text": self.documents[doc_id],
                    "score": float(1 / (1 + distances[0][i]))
                })
        return results

# make vector store instance
vector_store = VectorStore()

# all of the API endpoints
@app.get("/")
def read_root():
    return {"message": "Thank you for this opportunity and I hope you enjoy the RAG demo!"}

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

# load initial data
@app.on_event("startup")
def load_initial_data():
    for doc_data in DOCUMENTS:
        doc = Document(**doc_data)
        vector_store.add_document(doc)
    # print("data loaded")
