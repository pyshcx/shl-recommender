"""
test_api.py

Purpose:
    Simple script to test the FastAPI endpoints.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    print(f"Testing GET {BASE_URL}/health ...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print("Status Code:", response.status_code)
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        
        assert "status" in data
        assert data["status"] == "healthy"
        print("✅ Health check passed!\n")
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")

def test_recommend():
    endpoint = f"{BASE_URL}/recommend"
    print(f"Testing POST {endpoint} ...")
    
    payload = {
        "query": "software engineer python skills",
        "top_k": 3
    }
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print("Status Code:", response.status_code)
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        
        # Validate structure
        assert "query" in data
        assert "results" in data
        assert isinstance(data["results"], list)
        
        print("✅ Recommendation endpoint structure valid!\n")
    except Exception as e:
        print(f"❌ Recommendation test failed: {e}\n")

if __name__ == "__main__":
    test_health()
    test_recommend()
