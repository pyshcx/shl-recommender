import json
import re
import os
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_metadata(assessments):
    """
    Enhances the assessment dataset by extracting structured metadata 
    (duration, adaptive_support, test_type, remote_support) using Regex 
    formulas on the raw text description.
    """
    enhanced = []
    
    for item in assessments:
        desc = item.get("description", "")
        
        # We don't overwrite if the scraper already found explicit data,
        # but we add it if missing.
        
        # 1. Duration (e.g. "Approximate Completion Time in minutes = 15")
        if item.get("duration") is None and desc:
            duration_match = re.search(r'Approximate Completion Time in minutes\s*=\s*([^T]*?)(?=\s*Test Type:|$)', desc, re.IGNORECASE)
            if duration_match:
                item["duration"] = duration_match.group(1).strip()
                
        # 2. Adaptive Support
        if item.get("adaptive_support") is None and desc:
            if re.search(r'adaptive test|adaptive assessment', desc, re.IGNORECASE):
                item["adaptive_support"] = True
            else:
                item["adaptive_support"] = False
                
        # 3. Test Type Codes
        if item.get("test_type") is None and desc:
            test_type_match = re.search(r'Test Type:\s*([A-Za-z0-9]+)', desc, re.IGNORECASE)
            if test_type_match:
                item["test_type"] = test_type_match.group(1).strip()
                
        # 4. Remote Support
        if item.get("remote_support") is None and desc:
            if re.search(r'Remote Testing:', desc, re.IGNORECASE):
                item["remote_support"] = True
            else:
                item["remote_support"] = False

        enhanced.append(item)
        
    return enhanced

def main():
    parser = ArgumentParser(description="Clean and enhance the SHL assessments dataset metadata.")
    parser.add_argument("--input", default="data/processed/assessments_clean.json", help="Input JSON path")
    parser.add_argument("--output", default="data/processed/assessments_clean_enhanced.json", help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    logger.info(f"Loading {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Extracting metadata for {len(data)} assessments...")
    enhanced_data = extract_metadata(data)

    # Overwrite the clean file or save to a new one
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=4)
        
    logger.info(f"Saved enhanced dataset to {args.output}")

if __name__ == "__main__":
    main()
