"""
recall_eval.py

Purpose:
    Module to evaluate the retrieval system's performance using metrics like Recall@K.
"""

import os
import ast
import json
import logging
import argparse
import pandas as pd
from typing import List, Dict, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RecallEval")


def compute_recall_at_k(actual: List[str], predicted: List[str], k: int) -> float:
    """
    Computes the Recall@K metric given actual relevant items and predicted items.
    
    Args:
        actual (list): List of actual relevant item identifiers (e.g., URLs or IDs).
        predicted (list): List of predicted item identifiers from the retrieval system.
        k (int): The cutoff rank for computing recall.
        
    Returns:
        float: The Recall@K value.
    """
    if not actual:
        return 0.0

    # Truncate predicted list to top K items
    top_k_predicted = predicted[:k]
    
    # Determine the intersection between actual and truncated predicted items
    intersection = set(actual).intersection(set(top_k_predicted))
    
    # Calculate and return (intersection size) / (actual size)
    return len(intersection) / len(actual)


def load_actuals(labeled_path: str) -> Dict[str, List[str]]:
    """
    Loads labeled actual relevant URLs for each query.
    Assumes JSON mapping: { "query": ["url1", "url2", ...] } 
    or a DataFrame with 'query' and 'relevant_urls' columns.
    """
    logger.info(f"Loading actuals from {labeled_path}")
    if labeled_path.endswith('.json'):
        with open(labeled_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif labeled_path.endswith('.csv'):
        df = pd.read_csv(labeled_path)
        actuals = {}
        for _, row in df.iterrows():
            urls = row['relevant_urls']
            if isinstance(urls, str):
                try:
                    urls = ast.literal_eval(urls)
                except:
                    urls = [urls]
            actuals[row['query']] = urls
        return actuals
    else:
        raise ValueError("Labeled dataset must be .csv or .json")


def load_predictions(predictions_path: str) -> Dict[str, List[str]]:
    """
    Loads model predictions for each query.
    Assumes a CSV with 'query' and 'predicted_urls' (as a list string).
    """
    logger.info(f"Loading predictions from {predictions_path}")
    df = pd.read_csv(predictions_path)
    predictions = {}
    for _, row in df.iterrows():
        urls = row['predicted_urls']
        if isinstance(urls, str):
            try:
                urls = ast.literal_eval(urls)
            except:
                # Fallback if it's not a python list string, maybe comma separated
                urls = [u.strip() for u in urls.split(",")]
        predictions[row['query']] = urls
    return predictions


def evaluate(predictions_path: str, labeled_path: str, k: int = 10):
    """
    Evaluates predictions against labeled data using Recall@K.
    """
    if not os.path.exists(predictions_path):
        logger.error(f"Predictions file not found: {predictions_path}")
        return
        
    if not os.path.exists(labeled_path):
        logger.error(f"Labeled data file not found: {labeled_path}")
        return

    actuals = load_actuals(labeled_path)
    predictions = load_predictions(predictions_path)

    recalls = []
    
    print(f"\nEvaluating Recall@{k} per query:")
    print("-" * 50)
    
    for query, actual_urls in actuals.items():
        if query not in predictions:
            logger.warning(f"No prediction found for query: '{query}'")
            continue
            
        predicted_urls = predictions[query]
        recall = compute_recall_at_k(actual_urls, predicted_urls, k)
        recalls.append(recall)
        
        print(f"Query: '{query}'")
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  Actual: {len(actual_urls)} items, Predicted: {len(predicted_urls)} items")
        
    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        print("-" * 50)
        print(f"Mean Recall@{k}: {mean_recall:.4f} (across {len(recalls)} queries)\n")
    else:
        print("\nNo matching queries found to evaluate.\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Retrieval System Recall@K")
    parser.add_argument(
        "--predictions", 
        type=str, 
        required=True,
        help="Path to predictions CSV file (query, predicted_urls)."
    )
    parser.add_argument(
        "--labeled", 
        type=str, 
        required=True,
        help="Path to labeled data JSON/CSV (query, relevant_urls)."
    )
    parser.add_argument(
        "-k", 
        type=int, 
        default=10,
        help="K value for Recall@K (default: 10)."
    )
    
    args = parser.parse_args()
    evaluate(args.predictions, args.labeled, k=args.k)


if __name__ == "__main__":
    main()
