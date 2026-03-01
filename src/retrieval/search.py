"""
search.py

Purpose:
    Provides semantic retrieval functionality against the SHL assessments FAISS index using TF-IDF.
"""

import os
import pickle
import logging
import argparse
from typing import List, Dict, Any

import numpy as np

from src.retrieval.vector_store import FAISSIndexManager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SemanticSearch")


class AssessmentSearcher:
    """Handles semantic querying against the assessment index."""

    def __init__(self, index_dir: str = 'data/embeddings/'):
        self.index_dir = index_dir
        
        self.vectorizer_path = os.path.join(index_dir, "vectorizer.pkl")
        self.vectorizer = self._load_vectorizer()
        
        logger.info(f"Initializing FAISS IndexManager from {index_dir}")
        self.index_manager = FAISSIndexManager()
        self._load_index()

    def _load_vectorizer(self):
        """Loads the saved TF-IDF vectorizer."""
        if not os.path.exists(self.vectorizer_path):
            logger.error(f"Vectorizer not found at {self.vectorizer_path}")
            return None
            
        try:
            logger.info(f"Loading vectorizer: {self.vectorizer_path}")
            with open(self.vectorizer_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load vectorizer: {e}")
            return None

    def _load_index(self):
        """Loads the pre-built FAISS index and metadata."""
        if not self.index_manager.load_index(self.index_dir):
            logger.error("Failed to load FAISS index and/or metadata.")

    def _compute_relevance_score(self, l2_distance: float) -> float:
        """
        Convert L2 distance into a normalized relevance score (0.0 to 1.0).
        Since L2 distance bounds depend on the vectors, we use a decay function.
        score = 1 / (1 + distance) limits the result to (0, 1].
        """
        return 1.0 / (1.0 + float(l2_distance))

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k matching assessments for a given semantic query.
        Returns structured objects with name, url, score, and the full original assessment.
        """
        if not query.strip():
            logger.warning("Empty query provided.")
            return []
            
        if self.index_manager.index is None or not self.index_manager.metadata:
            logger.error("Index or metadata not loaded. Cannot perform search.")
            return []
            
        if self.vectorizer is None:
            logger.error("Vectorizer not loaded. Cannot generate embeddings.")
            return []

        logger.info(f"Embedding query: '{query}'")
        try:
            # Generate embedding for the query using TF-IDF vectorizer
            sparse_vector = self.vectorizer.transform([query])
            query_vector = sparse_vector.toarray().astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        # Ensure k is not larger than index size
        k_to_search = min(top_k, self.index_manager.index.ntotal)
        if k_to_search <= 0:
            return []
            
        logger.info(f"Searching index for top {k_to_search} matches...")
        
        try:
            # Get raw results from FAISS
            distances, indices = self.index_manager.index.search(query_vector, k_to_search)
        except Exception as e:
            logger.error(f"Error during FAISS search calculation: {e}")
            return []
        
        formatted_results = []
        
        # indices and distances are 2D arrays (1, k)
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 if there are no more neighbors
                continue
                
            dist = distances[0][i]
            relevance_score = self._compute_relevance_score(dist)
            
            record = self.index_manager.metadata[idx]
            
            # Structured object with specified requirements
            result_item = {
                "name": record.get("name", "Unknown"),
                "url": record.get("url", ""),
                "score": round(relevance_score, 4),
                "assessment": record  # Keep full record for downstream tasks (like reranking)
            }
            formatted_results.append(result_item)
            
        logger.info(f"Retrieved {len(formatted_results)} results.")
        return formatted_results


# Global cache for searcher instances to avoid reloading index/vectorizer
_global_searchers = {}

def retrieve(query: str, top_k: int = 20, index_dir: str = 'data/embeddings/') -> List[Dict[str, Any]]:
    """
    Standalone function to quickly run semantic retrieval.
    Uses singleton-style caching to ensure the FAISS index and vectorizer
    are only loaded once per application lifecycle to drastically reduce latency.
    """
    global _global_searchers
    
    if index_dir not in _global_searchers:
        logger.info(f"First time loading searcher for index: {index_dir}")
        _global_searchers[index_dir] = AssessmentSearcher(index_dir=index_dir)
        
    return _global_searchers[index_dir].retrieve(query, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="SHL Semantic Search")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True,
        help="The search query string."
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=20,
        help="Number of results to retrieve."
    )
    parser.add_argument(
        "--index-dir", 
        type=str, 
        default="data/embeddings/",
        help="Directory where faiss.index, metadata.json, and vectorizer.pkl are stored."
    )
    args = parser.parse_args()

    index_dir = os.path.abspath(args.index_dir)
    
    searcher = AssessmentSearcher(index_dir=index_dir)
    results = searcher.retrieve(args.query, top_k=args.top_k)
    
    print(f"\n--- Search Results for: '{args.query}' ---\n")
    for i, res in enumerate(results, 1):
        score = res.get('score', 0)
        name = res.get('name', 'Unknown')
        url = res.get('url', '')
        
        print(f"{i:2d}. [Score: {score:.4f}] {name}")
        print(f"    URL: {url}")
        print()


if __name__ == "__main__":
    main()
