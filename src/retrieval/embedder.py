"""
embedder.py

Purpose:
    Generates embeddings for SHL assessments using scikit-learn TfidfVectorizer.
    Provides a stable, CPU-only embedding baseline.
"""

import os
import json
import pickle
import logging
import argparse
import numpy as np
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Embedder")


def load_assessments(filepath: str) -> List[Dict[str, Any]]:
    """Loads assessment data from a JSON file."""
    if not os.path.exists(filepath):
        logger.error(f"Input file not found: {filepath}")
        return []
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} assessments from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        return []


def create_embeddings(
    input_file: str, 
    output_dir: str
):
    """Generates and saves TF-IDF embeddings, vectorizer, and metadata."""
    assessments = load_assessments(input_file)
    if not assessments:
        logger.warning("No data to embed. Aborting.")
        return

    # Prepare text for embedding and metadata
    texts_to_embed = []
    metadata = []
    
    for item in assessments:
        name = item.get("name", "")
        description = item.get("description", "")
        
        # Combine name and description
        text = f"{name} {description}".strip()
        texts_to_embed.append(text)
        
        # Keep original item as metadata
        metadata.append(item)

    logger.info("Initializing TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True
    )

    logger.info(f"Generating embeddings for {len(texts_to_embed)} items...")
    try:
        sparse_embeddings = vectorizer.fit_transform(texts_to_embed)
        # Convert to dense numpy array
        dense_embeddings = sparse_embeddings.toarray().astype(np.float32)
        logger.info(f"Generated dense embeddings of shape: {dense_embeddings.shape}")
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    try:
        np.save(embeddings_path, dense_embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")

    # Save vectorizer
    vectorizer_path = os.path.join(output_dir, "vectorizer.pkl")
    try:
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Saved vectorizer to {vectorizer_path}")
    except Exception as e:
        logger.error(f"Failed to save vectorizer: {e}")

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate TF-IDF embeddings for SHL assessments.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/processed/assessments_clean.json",
        help="Path to the input JSON file containing assessments."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/embeddings/",
        help="Directory to save the generated embeddings, vectorizer, and metadata."
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)

    create_embeddings(input_path, output_dir)


if __name__ == "__main__":
    main()
