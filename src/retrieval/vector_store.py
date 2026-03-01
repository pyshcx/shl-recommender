"""
vector_store.py

Purpose:
    Manages the FAISS vector index for fast similarity search.
"""

import os
import json
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple

import faiss

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VectorStore")


class FAISSIndexManager:
    """Manages creation, loading, and querying of a FAISS index."""

    def __init__(self):
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def build_index(self, embeddings_dir: str):
        """Builds a FAISS index from saved embeddings and metadata."""
        embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
        metadata_path = os.path.join(embeddings_dir, "metadata.json")

        if not os.path.exists(embeddings_path):
            logger.error(f"Embeddings file not found: {embeddings_path}")
            return
            
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            return

        try:
            logger.info(f"Loading embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
            
            # Ensure float32 for FAISS
            embeddings = embeddings.astype(np.float32)
            
            logger.info(f"Loaded embeddings matrix of shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return

        try:
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            if len(self.metadata) != embeddings.shape[0]:
                logger.error("Mismatch between number of embeddings and metadata records!")
                return
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return

        # Build IndexFlatL2
        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatL2 with dimension {dimension}...")
        try:
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            logger.info(f"Successfully added {self.index.ntotal} vectors to FAISS index.")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return

        # Save the index
        index_path = os.path.join(embeddings_dir, "faiss.index")
        try:
            faiss.write_index(self.index, index_path)
            logger.info(f"Successfully saved FAISS index to {index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def load_index(self, index_dir: str) -> bool:
        """Loads a pre-built FAISS index and metadata."""
        index_path = os.path.join(index_dir, "faiss.index")
        metadata_path = os.path.join(index_dir, "metadata.json")

        if not os.path.exists(index_path):
            logger.error(f"FAISS index file not found: {index_path}")
            return False

        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            return False

        try:
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            if self.index.ntotal != len(self.metadata):
                logger.warning(f"Warning: Index contains {self.index.ntotal} vectors but metadata has {len(self.metadata)} records.")
                
            logger.info("Successfully loaded FAISS index and metadata.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index or metadata: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Manage FAISS Vector Index.")
    parser.add_argument(
        "--action", 
        type=str, 
        choices=["build"],
        default="build",
        help="Action to perform: 'build' to create a new index from embeddings."
    )
    parser.add_argument(
        "--embeddings-dir", 
        type=str, 
        default="data/embeddings/",
        help="Directory containing embeddings.npy and where faiss.index will be saved."
    )
    args = parser.parse_args()

    embeddings_dir = os.path.abspath(args.embeddings_dir)
    
    manager = FAISSIndexManager()
    
    if args.action == "build":
        manager.build_index(embeddings_dir)


if __name__ == "__main__":
    main()
