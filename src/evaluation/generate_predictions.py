"""
generate_predictions.py

Purpose:
    Runs the recommendation pipeline (retrieve + rerank) on sample queries
    and saves the resulting recommendations format to a CSV for offline evaluation.
"""

import os
import csv
import sys
import logging
import argparse
from typing import List

# Add project root to path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dotenv import load_dotenv

# Load environment configuration (specifically for Gemini api key)
load_dotenv()

from src.retrieval.search import retrieve
from src.reranker.rerank import rerank

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Evaluator")

def generate_predictions(queries: List[str], output_csv: str):
    """
    Given a list of test queries, runs the search + rerank pipeline and 
    saves the top 10 recommended URLs to a CSV file.
    """
    logger.info(f"Generating predictions for {len(queries)} queries.")
    
    # Track the final rows to write to CSV
    results_rows = []

    for idx, query in enumerate(queries, 1):
        logger.info(f"Processing query {idx}/{len(queries)}: '{query}'")
        
        try:
            # 1. Retrieve Candidate Pool (Default 20)
            candidates = retrieve(query, top_k=20)
            
            # 2. Rerank using Gemini (Expect up to 10 returned) 
            # We enforce returning the top 10 URLs specifically.
            if candidates:
                final_rankings = rerank(query, candidates)
            else:
                final_rankings = []
            
            # 3. Extract purely the URLs for the evaluation set and enforce 10 max
            top_10_urls = [res.get("url", "") for res in final_rankings][:10]
            
            # Format requires one row per recommendation
            if not top_10_urls:
                results_rows.append({
                    "Query": query,
                    "Assessment_url": ""
                })
            else:
                for url in top_10_urls:
                    results_rows.append({
                        "Query": query,
                        "Assessment_url": url
                    })
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            # Insert a blank result so we don't offset evaluation rows
            results_rows.append({
                "Query": query,
                "Assessment_url": ""
            })

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Query", "Assessment_url"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in results_rows:
                writer.writerow(row)
                
        logger.info(f"Successfully saved predictions to {output_csv}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Recommendation Predictions")
    parser.add_argument(
        "--input-file", 
        type=str, 
        default=None,
        help="Path to a text file containing one query per line. If omitted, uses default sample queries."
    )
    parser.add_argument(
        "--output-csv", 
        type=str, 
        default="data/evaluation/predictions.csv",
        help="Output CSV path."
    )
    args = parser.parse_args()

    # Pre-defined sample testing queries if no file is provided
    default_queries = [
        "Python developer with strong communication skills",
        "React and Node.js fullstack engineer",
        "entry level customer service and typing skills",
        "senior data scientist with machine learning expertise",
        "Java backend developer who understands teamwork"
    ]

    queries_to_run = []
    if args.input_file and os.path.exists(args.input_file):
        logger.info(f"Loading queries from file: {args.input_file}")
        if args.input_file.endswith('.xlsx'):
            import pandas as pd
            df = pd.read_excel(args.input_file, sheet_name='Test-Set')
            # Extract queries and convert to string, dropping NaNs
            queries_to_run = [str(q) for q in df['Query'].dropna().tolist()]
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                queries_to_run = [line.strip() for line in f if line.strip()]
    else:
        logger.info("Using default sample queries.")
        queries_to_run = default_queries

    generate_predictions(
        queries=queries_to_run,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main()
