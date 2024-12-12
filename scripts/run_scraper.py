#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_loader.enhanced_scraper import EnhancedScraper

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scraper.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Initialize scraper
    output_dir = project_root / "datasets" / "processed"
    scraper = EnhancedScraper(output_dir=str(output_dir))

    # List of drugs to scrape
    drug_list = [
        "Amoxicillin",
        "Lisinopril",
        "Metformin",
        "Omeprazole",
        "Simvastatin"
    ]

    # Scrape drugs one by one with delays
    for drug in drug_list:
        try:
            logger.info(f"Starting scrape for {drug}")
            result = scraper.scrape_drug(drug)
            
            if result:
                logger.info(f"Successfully scraped {drug}")
            else:
                logger.warning(f"Failed to scrape {drug}")
            
            # Add a significant delay between requests to avoid rate limiting
            logger.info("Waiting 5 minutes before next scrape to avoid rate limits...")
            time.sleep(300)  # 5 minutes between requests
        
        except Exception as e:
            logger.error(f"Unexpected error scraping {drug}: {str(e)}")
            # Wait longer if there's an error
            time.sleep(600)  # 10 minutes

    logger.info("Scraping process completed")

if __name__ == "__main__":
    main()
