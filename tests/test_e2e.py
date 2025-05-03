import json
from fastapi import Response
from fastapi.testclient import TestClient
import pandas as pd
import os
from main import app

client = TestClient(app)

csv_path = os.path.join(os.path.dirname(__file__), "./chunks_test.csv")
test_chunks_df = pd.read_csv(csv_path)

create_library_response: Response | None = None

def create_library(client: TestClient) -> Response:
    document_names = test_chunks_df["document_name"].unique().tolist()
    documents = [
        {
            "title": document_name,
            "description": f"Test document for {document_name}",
            "metadata": {
                "source": "string",
                "tags": ["test"]
            },
            "chunks": [
                {
                    "text": chunk,
                    "metadata": {
                        "source": "string",
                        "tags": ["test"]
                    }
                }
                for chunk in test_chunks_df[test_chunks_df["document_name"] == document_name]["chunk"].tolist()
            ] 
        } 
        for document_name in document_names
    ]

    library_request = {
        "name": "Test Library",
        "metadata": {
            "created_at": "2025-05-02T10:04:47.536Z",
            "updated_at": "2025-05-02T10:04:47.536Z",
            "tags": [
            "test", "default"
            ]
        },
        "indexer": "hsnw",
        "documents": documents
    }
    response = client.post("/libraries", json=library_request)
    return response

def test_create_library():
    global create_library_response
    create_library_response = create_library(client)
    assert create_library_response.status_code == 200
    assert create_library_response.json()["name"] == "Test Library"

def test_search_library():
    global create_library_response
    assert create_library_response.status_code == 200
    assert create_library_response.json()["name"] == "Test Library"
    library_id = create_library_response.json()["library_id"]
    search_request = {
        "query": "How do I get started with crypto mining?",
        "k": 5,
        "distance_metric": "cosine",
        "include_metadata": False,
        "include_embeddings": False
    }
    search_response = client.post(f"/libraries/{library_id}/search", json=search_request)
    assert search_response.status_code == 200
    # Assert that the response contains the expected fields
    assert "query" in search_response.json()
    assert "results" in search_response.json()
    assert "documents" in search_response.json()
    assert "total_results" in search_response.json()
    
    # Assert that we have results
    assert len(search_response.json()["results"]) > 0
    
    # Check for the specific chunk text in the results
    expected_chunk_text = "Mining for cryptocurrencies like Bitcoin consumes significant amounts of electricity and may be contributing to global warming. Some cryptocurrencies are, by design, not reliant on this type of processing power."
    
    # Find if any result contains the expected text
    found_expected_text = False
    for result in search_response.json()["results"]:
        if result["text"] == expected_chunk_text:
            found_expected_text = True
            break

    print(json.dumps([x["text"] for x in search_response.json()["results"]], indent=4))
    
    # Assert that the expected chunk text was found in the results
    assert found_expected_text, "Expected chunk text not found in search results"