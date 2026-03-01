"""
SHL Scraper module.
Two-stage crawler for SHL product catalog.
Stage 1: Link discovery
Stage 2: Detail scraping
"""

import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Set, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SHLScraper")


class SHLScraper:
    """Two-stage crawler for SHL assessments."""
    
    BASE_URL = "https://www.shl.com"
    CATALOG_URL = "https://www.shl.com/products/product-catalog/"

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = self._setup_session()
        self.results: List[Dict[str, Any]] = []

    def _setup_session(self) -> requests.Session:
        """Configure requests session with robust retry logic."""
        session = requests.Session()
        
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Standard headers to avoid anti-bot blocks
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        
        return session

    def fetch_page(self, url: str) -> Optional[requests.Response]:
        """Fetch a URL with error handling and delay."""
        try:
            logger.info(f"Fetching {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            time.sleep(self.delay)  # Polite delay
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def discover_links(self, start_url: str = CATALOG_URL) -> Set[str]:
        """Stage 1: Discover all product links via pagination."""
        discovered_urls: Set[str] = set()
        start_index = 0
        step = 12
        
        logger.info("Starting Stage 1: Link Discovery...")
        
        while True:
            # Construct pagination URL
            parsed_url = urlparse(start_url)
            query_params = parse_qs(parsed_url.query)
            query_params['start'] = [str(start_index)]
            
            new_query = urlencode(query_params, doseq=True)
            current_url = urlunparse(parsed_url._replace(query=new_query))
            
            logger.info(f"Scanning catalog pagination start={start_index}")
            response = self.fetch_page(current_url)
            
            if not response:
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            found_on_page = 0
            for link in links:
                href = link['href']
                if '/products/product-catalog/view/' in href:
                    full_url = urljoin(self.BASE_URL, href)
                    if full_url not in discovered_urls:
                        discovered_urls.add(full_url)
                        found_on_page += 1
                        
            # If no new links matching the product view are found, we assume we've hit the end
            if found_on_page == 0:
                logger.info("No more product links found on page. Ending discovery.")
                break
                
            start_index += step
            
        logger.info(f"Stage 1 Complete. Discovered {len(discovered_urls)} unique product URLs.")
        return discovered_urls

    def scrape_details(self, urls: Set[str]) -> List[Dict[str, Any]]:
        """Stage 2: Visit each product page and extract details."""
        logger.info("Starting Stage 2: Detail Scraping...")
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Scraping product {i}/{len(urls)}: {url}")
            response = self.fetch_page(url)
            if not response:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract basic details
            name = "Unknown"
            if soup.title:
                name = soup.title.get_text(strip=True).replace(" | SHL", "")
                
            # Attempt to find main body text for description
            # E.g. find all paragraph tags or main content block
            description_paragraphs = soup.find_all('p')
            description = " ".join([p.get_text(strip=True) for p in description_paragraphs if p.get_text(strip=True)])
            
            # Extract spec details if present in the text
            duration = None
            adaptive_support = None
            remote_support = None
            test_type = None
            
            text_lower = soup.get_text(separator=' ', strip=True).lower()
            
            if "adaptive" in text_lower:
                adaptive_support = True
            if "remote" in text_lower or "online" in text_lower:
                remote_support = True
                
            # Naive heuristics for duration and type, usually found in spec lists
            list_items = soup.find_all('li')
            for li in list_items:
                li_text = li.get_text(strip=True).lower()
                if "min" in li_text or "duration" in li_text:
                    duration = li.get_text(strip=True)
                elif "type:" in li_text:
                    test_type = li.get_text(strip=True).replace("type:", "", 1).strip()
                    
            item_data = {
                "name": name,
                "url": url,
                "description": description
            }
            if duration is not None: item_data["duration"] = duration
            if adaptive_support is not None: item_data["adaptive_support"] = adaptive_support
            if remote_support is not None: item_data["remote_support"] = remote_support
            if test_type is not None: item_data["test_type"] = test_type
            
            self.results.append(item_data)
            
        logger.info(f"Stage 2 Complete. Successfully scraped {len(self.results)} items.")
        return self.results

    def save_results(self, output_path: str) -> None:
        """Save extracted results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results successfully saved to: {output_path}")
        except IOError as e:
            logger.error(f"Failed to save results to {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="SHL Catalog Scraper (Two-Stage)")
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/assessments.json",
        help="Path where the JSON output should be saved."
    )
    parser.add_argument(
        "--start-url", 
        type=str, 
        default="https://www.shl.com/products/product-catalog/",
        help="Starting Base URL for the catalog."
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.output)
    
    logger.info("Initializing SHL Scraper...")
    scraper = SHLScraper()
    
    # Stage 1
    discovered_urls = scraper.discover_links(start_url=args.start_url)
    
    if discovered_urls:
        # Stage 2
        scraper.scrape_details(discovered_urls)
        scraper.save_results(out_path)
    else:
        logger.warning("No URLs discovered. Skipping Stage 2 and save.")
        
    logger.info("Successfully finished scraping workflow.")


if __name__ == "__main__":
    main()
