# Civic Backend Project

## Features

- Store text documents with unique IDs
- Convert text to vector embeddings using Sentence Transformers
- Index and search vectors efficiently with FAISS
- Query documents by semantic similarity
- Simple REST API with FastAPI

## Installation

1. Clone this repository:
git clone https://github.com/kevinlindong/civic_backend_project.git
cd simple-rag-api

2. Install the required packages:
pip install -r requirements.txt

## Testing

Run the tests with:
pytest test_app.py -v
Make sure the API server is running before executing the tests by running "uvicorn app --reload"