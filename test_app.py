# imports
import pytest
import requests

@pytest.fixture
def api_url():
    """Fixture to provide the base API URL."""
    return "http://localhost:8000"

def test_ingest(api_url):
    """Test the /ingest endpoint with a valid document."""
    test_doc = {
        "id": 8,
        "text": "Python is a popular programming language."
    }
    response = requests.post(f"{api_url}/ingest", json=test_doc)
    
    #check response
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"

def test_query_basic(api_url):
    "Test the /query endpoint for relevant results."
    # query for documents about Civic
    response = requests.get(f"{api_url}/query?text=What is Civic?&top_k=2")
    
    # check response
    assert response.status_code == 200

    # check if any results are returned
    results = response.json()
    assert len(results) > 0
    
    # check if any result mentions Civic
    civic_found = any("Civic" in result["text"] for result in results)
    assert civic_found

def test_query_multiple_results(api_url):
    "Test the /query endpoint with multiple results."
    # query for documents about California which matches multiple documents
    response = requests.get(f"{api_url}/query?text=California&top_k=3")
    
    # check response
    assert response.status_code == 200

    # check if multiple results are returned
    results = response.json()
    assert len(results) > 1