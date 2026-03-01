"""
evaluate.py

Purpose:
    Evaluates the SHL recommender module by computing Recall@10 against a 
    provided ground truth dataset.
"""

import os
import sys
import json
import logging
import argparse

# Add project root to path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.search import retrieve
from src.reranker.rerank import rerank

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Evaluator")

def evaluate(ground_truth_file: str):
    """
    Evaluates Recall@10 for the recommender pipeline.
    Expects a JSON file with a list of dictionaries:
    [
      {
        "query": "Java developer",
        "expected_urls": ["url1", "url2", ...]
      },
      ...
    ]
    """
    if not os.path.exists(ground_truth_file):
        logger.error(f"Ground truth file not found: {ground_truth_file}")
        return

    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load ground truth data: {e}")
        return

    total_queries = len(test_data)
    if total_queries == 0:
        logger.warning("Ground truth file is empty.")
        return

    correct_matches = 0

    logger.info(f"Starting evaluation on {total_queries} queries...")

    for idx, item in enumerate(test_data, 1):
        query = item.get("query", "")
        expected_urls = set(item.get("expected_urls", []))
        
        if not query or not expected_urls:
            logger.warning(f"Skipping row {idx} due to missing query or expected_urls.")
            continue
            
        logger.info(f"Evaluating query {idx}/{total_queries}: '{query}'")

        try:
            # 1. Retrieve candidates
            candidates = retrieve(query, top_k=20)
            
            # 2. Rerank 
            final_rankings = rerank(query, candidates) if candidates else []
            
            # 3. Extract top 10 URLs
            top_10_urls = [res.get("url", "") for res in final_rankings][:10]
            
            # 4. Check for any match (Recall@10 definition here: at least one relevant item found)
            # If standard Recall@10 means finding ALL relevant, this is effectively hit rate.
            # Assuming "Check if any predicted URL appears in ground truth list" from requirements.
            match_found = any(url in expected_urls for url in top_10_urls)
            
            if match_found:
                correct_matches += 1
                
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            continue

    recall_at_10 = correct_matches / total_queries

    print("\n" + "="*40)
    print("EVALUATION RESULTS (Recall@10)")
    print("="*40)
    print(f"Total queries:   {total_queries}")
    print(f"Correct matches: {correct_matches}")
    print(f"Recall@10 score: {recall_at_10:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Recommender Recall@10")
    parser.add_argument(
        "--ground-truth", 
        type=str, 
        required=True,
        help="Path to JSON file containing ground truth queries and expected URLs."
    )
    args = parser.parse_args()
    
    evaluate(args.ground_truth)
