"""
Dataset processing module.
Cleans and normalizes the scraped SHL assessments dataset.
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DatasetBuilder")


class DatasetBuilder:
    """Processes and cleans the raw scraped dataset."""

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.raw_data: List[Dict[str, Any]] = []
        self.clean_data: List[Dict[str, Any]] = []

    def load_data(self) -> None:
        """Load raw dataset from JSON."""
        if not os.path.exists(self.input_path):
            logger.error(f"Input file not found: {self.input_path}")
            return
            
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            logger.info(f"Loaded {len(self.raw_data)} records from {self.input_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {self.input_path}: {e}")
        except IOError as e:
            logger.error(f"Failed to read file {self.input_path}: {e}")

    def clean_text(self, text: str) -> str:
        """Clean and normalize a text field."""
        if not text:
            return ""
        # Remove excess whitespace and normalize spaces
        return " ".join(text.split()).strip()

    def process_data(self) -> None:
        """Clean text fields, normalize types, and remove duplicates."""
        seen_names = set()
        
        for item in self.raw_data:
            # Clean essential fields
            name = self.clean_text(item.get("name", ""))
            description = self.clean_text(item.get("description", ""))
            
            # Ensure name and description exist
            if not name or not description:
                continue
                
            # Deduplication based on name
            normalized_name = name.lower()
            if normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)
            
            # Clean other text fields
            url = self.clean_text(item.get("url", ""))
            duration = self.clean_text(item.get("duration", "Unknown"))
            
            # Normalize test_type to list
            test_type_raw = item.get("test_type", "")
            if isinstance(test_type_raw, str):
                # Split by comma or slash if multiple types exist
                types = [self.clean_text(t) for t in test_type_raw.replace("/", ",").split(",") if t.strip()]
                test_type = types if types else ["Standard"]
            elif isinstance(test_type_raw, list):
                test_type = [self.clean_text(str(t)) for t in test_type_raw if str(t).strip()]
            else:
                test_type = ["Standard"]
                
            clean_item = {
                "name": name,
                "url": url,
                "description": description,
                "duration": duration,
                "adaptive_support": bool(item.get("adaptive_support", False)),
                "remote_support": bool(item.get("remote_support", False)),
                "test_type": test_type
            }
            
            self.clean_data.append(clean_item)
            
        logger.info(f"Processed data: {len(self.clean_data)} valid records remaining after cleaning.")

    def save_data(self) -> None:
        """Save cleaned dataset to JSON."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.clean_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cleaned dataset saved successfully to: {self.output_path}")
        except IOError as e:
            logger.error(f"Failed to save results to {self.output_path}: {e}")

    def run(self) -> None:
        """Execute the full processing pipeline."""
        self.load_data()
        if self.raw_data:
            self.process_data()
            self.save_data()
        else:
            logger.warning("No data to process. Pipeline aborted.")


def main():
    parser = argparse.ArgumentParser(description="SHL Dataset Processor")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/processed/assessments.json",
        help="Path to the raw scraped JSON dataset."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/assessments_clean.json",
        help="Path where the cleaned JSON dataset should be saved."
    )
    args = parser.parse_args()

    # Determine absolute paths
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    
    logger.info("Initializing Dataset Builder...")
    builder = DatasetBuilder(input_path=input_path, output_path=output_path)
    builder.run()


if __name__ == "__main__":
    main()
