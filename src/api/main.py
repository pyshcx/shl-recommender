"""
main.py

Purpose:
    FastAPI application to serve the SHL assessment recommender system.
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

from src.retrieval.search import retrieve
from src.reranker.rerank import rerank

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("API")

app = FastAPI(
    title="SHL Recommender API",
    description="API for semantic search and LLM-based reranking of SHL assessments.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    logger.info("Health check requested.")
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: QueryRequest):
    """
    Given a search query:
    1. Retrieves top candidate assessments via TF-IDF / FAISS.
    2. Reranks the candidates using the Gemini LLM.
    3. Returns the final curated list of recommendations with full metadata.
    """
    logger.info(f"Received recommendation request for query: '{request.query}'")
    
    if not request.query.strip():
        logger.warning("Empty query received.")
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        # Step 1: Semantic Retrieval
        logger.info(f"Fetching initial candidates for: '{request.query}'")
        candidates = retrieve(request.query)
        
        if not candidates:
            logger.info("No candidates found during retrieval.")
            return {"query": request.query, "recommendations": []}

        # Step 2: LLM Reranking
        logger.info(f"Reranking {len(candidates)} candidates via LLM...")
        final_results = rerank(request.query, candidates)

        # Step 3: Enhance results with full metadata and regex extraction for missing fields
        import re
        enhanced_results = []
        candidate_lookup = {c["url"]: c.get("assessment", c) for c in candidates}
        
        for res in final_results:
            url = res.get("url")
            base_info = candidate_lookup.get(url, {})
            description = base_info.get("description", "")
            
            duration = base_info.get("duration")
            adaptive = base_info.get("adaptive_support")
            remote = base_info.get("remote_support")
            test_type = base_info.get("test_type")
            
            if duration is None and description:
                duration_match = re.search(r'Approximate Completion Time in minutes\s*=\s*([^T]*?)(?=\s*Test Type:|$)', description, re.IGNORECASE)
                if duration_match:
                    duration = duration_match.group(1).strip()
                    
            if adaptive is None and description:
                if re.search(r'adaptive test|adaptive assessment', description, re.IGNORECASE):
                    adaptive = True
                    
            if remote is None and description:
                if re.search(r'Remote Testing:', description, re.IGNORECASE):
                    remote = True
                    
            if test_type is None and description:
                test_type_match = re.search(r'Test Type:\s*([A-Za-z0-9]+)', description, re.IGNORECASE)
                if test_type_match:
                    test_type = test_type_match.group(1).strip()

            enhanced_results.append({
                "name": res.get("name", base_info.get("name", "Unknown")),
                "url": url,
                "description": description,
                "duration": duration,
                "adaptive_support": adaptive,
                "remote_support": remote,
                "test_type": test_type
            })

        logger.info(f"Returning {len(enhanced_results)} final recommendations.")
        return {
            "query": request.query,
            "recommendations": enhanced_results
        }
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing the recommendation.")

if __name__ == "__main__":
    import uvicorn
    # Typically run via `uvicorn src.api.main:app --reload`
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
