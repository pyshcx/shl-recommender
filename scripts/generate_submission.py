"""
generate_submission.py

Purpose:
    Runs the recommendation pipeline on a set of queries and outputs a CSV
    submission file in the format expected for scoring:
    
    Query,Assessment_url
"""

import os
import sys
import csv
import json
import logging
import argparse

# Add project root to path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.search import retrieve
from src.reranker.rerank import rerank

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SubmissionGenerator")

def generate_submission(input_queries_file: str, output_csv: str):
    if not os.path.exists(input_queries_file):
        logger.error(f"Input file not found: {input_queries_file}")
        return

    # Try to load as JSON (like ground truth) or flat text file
    queries = []
    try:
        with open(input_queries_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('[') or content.startswith('{'):
                data = json.loads(content)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    queries = [item.get("query", "") for item in data if "query" in item]
                elif isinstance(data, list):
                    queries = data
            else:
                queries = [line.strip() for line in content.split('\n') if line.strip()]
    except Exception as e:
        logger.error(f"Error reading queries file: {e}")
        return

    if not queries:
        logger.warning("No queries found.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    logger.info(f"Generating submission for {len(queries)} queries...")
    
    rows = []
    for idx, query in enumerate(queries, 1):
        logger.info(f"Processing query {idx}/{len(queries)}: '{query}'")
        try:
            candidates = retrieve(query, top_k=20)
            final_rankings = rerank(query, candidates) if candidates else []
            
            # Each recommended URL gets its own row
            for res in final_rankings:
                url = res.get("url", "")
                if url:
                    rows.append({
                        "Query": query,
                        "Assessment_url": url
                    })
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            continue

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Query", "Assessment_url"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        logger.info(f"Successfully generated submission file: {output_csv}")
    except Exception as e:
        logger.error(f"Failed to write CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Submission CSV")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to file containing queries (can be TXT with one query per line, or JSON array)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="submission.csv",
        help="Path to the output submission CSV file."
    )
    args = parser.parse_args()
    
    generate_submission(args.input, args.output)
