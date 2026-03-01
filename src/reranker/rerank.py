"""
rerank.py

Purpose:
    Module to re-rank the initially retrieved documents using a powerful LLM (Gemini API).
"""

import os
import json
import time
import logging
from typing import List, Dict, Any

import google.generativeai as genai
from google.api_core import retry

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LLMReranker")


def configure_gemini():
    """Configures the Gemini API client using the GEMINI_API_KEY environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Reranking will fail.")
        return False
        
    genai.configure(api_key=api_key)
    return True

# Simple retry logic for the LLM call to handle transient errors
@retry.Retry(initial=1.0, maximum=10.0, multiplier=2.0)
def _call_gemini_api(prompt: str) -> str:
    """Helper to call Gemini API with exponential backoff on retryable errors."""
    model = genai.GenerativeModel(
        'gemini-2.5-flash', 
        generation_config={"response_mime_type": "application/json"}
    )
    response = model.generate_content(prompt)
    return response.text


def rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-ranks a list of candidate assessments based on the query using the Gemini API.
    Enforces relevance, diversity, and balance of skill categories, returning 5-10 results.
    
    Args:
        query (str): The search query.
        candidates (list): List of initial search results (dictionaries).
        
    Returns:
        list: The re-ranked list of up to 10 assessments in structured JSON form ({name, url}).
    """
    if not candidates:
        logger.warning("No candidates provided for reranking.")
        return []

    if not configure_gemini():
        logger.warning("Returning top up to 10 original candidates due to missing API key.")
        return [{"name": c.get("name"), "url": c.get("url")} for c in candidates[:10]]

    # Limit the number of candidates sent to the LLM to avoid exceeding context limits
    # and improve response speed/quality.
    candidates_to_rank = candidates[:30]
    
    # We only need to send the relevant context to the LLM
    slim_candidates = []
    for c in candidates_to_rank:
        assessment = c.get("assessment", c)
        slim_candidates.append({
            "name": assessment.get("name", "Unknown"),
            "url": assessment.get("url", ""),
            "description": assessment.get("description", "")
        })
    
    candidates_json = json.dumps(slim_candidates, indent=2)

    prompt = f"""
    You are an expert HR, test recommendation, and assessment system.
    A user has searched for assessments with the following query: "{query}"

    Below is a list of candidate assessments retrieved from a semantic search.
    Your task is to review these candidates, rerank them, and select the best 5 to 10 assessments that match the query.

    When selecting and ordering the assessments, you MUST strictly adhere to the following criteria:
    1. Relevance: The assessment must primarily and directly address the skills, roles, or concepts mentioned in the query.
    2. Diversity: Avoid selecting identical tests sequentially. Introduce different styles of assessments if applicable, especially when the query spans multiple domains.
    3. Distinctiveness: Avoid duplicates or near-duplicates.
    4. Clarity: Prefer tests with clear, detailed descriptions.
    5. Size Limit: The final output must contain between 5 and 10 candidate assessments. Order them from most relevant to least relevant.

    Candidates data:
    {candidates_json}

    Return your response strictly as a JSON array of objects. 
    Each object in the array MUST contain ONLY the "name" and "url" fields matching the provided candidates.
    """

    logger.info(f"Sending prompt to Gemini API for reranking {len(slim_candidates)} candidates...")

    try:
        response_text = _call_gemini_api(prompt)
        
        # Parse the structured JSON response
        ranked_results = json.loads(response_text)
        
        # Handle cases where the model returns a slightly different structure (e.g. nested in a dict)
        if isinstance(ranked_results, dict) and len(ranked_results) == 1:
            key = list(ranked_results.keys())[0]
            if isinstance(ranked_results[key], list):
                ranked_results = ranked_results[key]
                
        if isinstance(ranked_results, list):
            # Enforce the strict limit programmatically as a fallback priority
            final_results = ranked_results[:10]
            logger.info(f"LLM successfully reranked and returned {len(final_results)} results.")
            return final_results
        else:
            logger.error("Unexpected JSON structure from Gemini model. Expected a list.")
            return [{"name": c.get("name"), "url": c.get("url")} for c in candidates[:10]]

    except Exception as e:
        logger.error(f"Error during Gemini reranking: {e}")
        # Fallback to the original ranking
        return [{"name": c.get("name"), "url": c.get("url")} for c in candidates[:10]]
