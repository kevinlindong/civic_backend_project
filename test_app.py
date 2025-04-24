# test_app.py
import pytest
import requests

@pytest.fixture
def api_url():
    """Fixture to provide the base API URL."""
    return "http://localhost:8000"

def test_ingest(api_url):
    """Test the /ingest endpoint with a valid document."""
    # Create a test document
    test_doc = {
        "id": 8,
        "text": "Python is a popular programming language."
    }
    
    # Send it to the API
    response = requests.post(f"{api_url}/ingest", json=test_doc)
    
    # Check the response
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"

def test_query_basic(api_url):
    """Test the /query endpoint for relevant results."""
    # Query for documents about Civic
    response = requests.get(f"{api_url}/query?text=What is Civic?&top_k=2")
    
    # Check the response
    assert response.status_code == 200
    
    results = response.json()
    assert len(results) > 0
    
    # Check if any result mentions Civic
    civic_found = any("Civic" in result["text"] for result in results)
    assert civic_found

def test_query_multiple_results(api_url):
    """Test the /query endpoint with multiple results."""
    # Query for documents about California (should match multiple documents)
    response = requests.get(f"{api_url}/query?text=California&top_k=3")
    
    # Check the response
    assert response.status_code == 200
    
    results = response.json()
    # We should get multiple results about California
    assert len(results) > 1