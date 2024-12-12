import json
import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MicroLabsScraper:
    BASE_URL = "https://www.microlabsusa.com"
    PRODUCTS_URL = f"{BASE_URL}/products/"
    DATASETS_PATH = "./datasets/microlabs_usa"

    def __init__(self):
        """Initialize the scraper with necessary session and setup"""
        self.session = requests.Session()
        self.products_data = {}
        os.makedirs(self.DATASETS_PATH, exist_ok=True)

    def get_all_product_urls(self) -> Dict[str, str]:
        """
        Fetch all product URLs from all pages of the products section
        Returns: Dictionary mapping product names to their URLs
        """
        logging.info("Fetching all product URLs...")
        products = {}
        current_url = self.PRODUCTS_URL
        page = 1

        while True:
            logging.info(f"Fetching page {page}...")
            response = self.session.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all product links in the current page
            found_products = False
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/products/' in href and href != self.PRODUCTS_URL and 'page' not in href:
                    product_name = link.get_text().strip()
                    if product_name and product_name != '« Older Entries' and product_name != 'Newer Entries »':
                        products[product_name] = href
                        found_products = True
            
            # Find next page link
            next_page = None
            for link in soup.find_all('a', href=True):
                if link.get_text().strip() == '« Older Entries':
                    next_page = link['href']
                    break
            
            if not next_page or not found_products:
                break
                
            current_url = next_page
            page += 1
            
            # Add a small delay to be respectful to the server
            time.sleep(1)
        
        logging.info(f"Found {len(products)} products across {page} pages")
        return products

    def get_prescribing_info_url(self, product_url: str) -> Optional[str]:
        """Extract the prescribing information URL from a product page"""
        response = self.session.get(product_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for h2 in soup.find_all('h2'):
            if h2.get_text().strip().lower() == "prescribing information":
                link = h2.find('a', href=True)
                if link:
                    return link['href']
        return None

    def extract_section_content(self, section_soup) -> str:
        """Extract content from a section, handling tables and lists"""
        content = []
        
        for element in section_soup.find_all(['p', 'table', 'ul', 'ol']):
            if element.name == 'table':
                # Handle tables
                table_content = []
                for row in element.find_all('tr'):
                    cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    table_content.append(" | ".join(cells))
                content.append("\n".join(table_content))
            elif element.name in ['ul', 'ol']:
                # Handle lists
                items = [item.get_text(strip=True) for item in element.find_all('li')]
                content.append("\n".join(f"- {item}" for item in items))
            else:
                # Handle paragraphs
                text = element.get_text(strip=True)
                if text:
                    content.append(text)
        
        return "\n\n".join(content)

    def parse_prescribing_info(self, url: str) -> Dict:
        """Parse the prescribing information page and extract structured data"""
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        info = {
            "indications_and_usage": "",
            "dosage_and_administration": "",
            "dosage_forms_and_strengths": "",
            "contraindications": "",
            "warnings_and_precautions": "",
            "adverse_reactions": "",
            "drug_interactions": "",
            "use_in_specific_populations": "",
            "description": "",
            "clinical_pharmacology": "",
            "how_supplied": "",
            "other_information": ""  # Catch-all for other sections
        }
        
        # First, try to find sections by exact heading matches
        for key in info.keys():
            section_name = key.replace("_", " ").title()
            section = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3', 'h4'] and 
                              section_name.lower() in tag.get_text().strip().lower())
            if section:
                content = []
                for sibling in section.find_next_siblings():
                    if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                        break
                    content.append(sibling)
                
                if content:
                    temp_soup = BeautifulSoup("<div></div>", "html.parser")
                    for elem in content:
                        temp_soup.div.append(elem)
                    info[key] = self.extract_section_content(temp_soup.div)
        
        # Then try to find sections by partial matches
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            heading_text = tag.get_text().strip().lower()
            
            # Map common variations to our keys
            mapping = {
                'indication': 'indications_and_usage',
                'dosage': 'dosage_and_administration',
                'dosage form': 'dosage_forms_and_strengths',
                'warning': 'warnings_and_precautions',
                'adverse': 'adverse_reactions',
                'interaction': 'drug_interactions',
                'specific population': 'use_in_specific_populations',
                'clinical pharmacology': 'clinical_pharmacology',
                'how supplied': 'how_supplied',
                'storage': 'how_supplied'
            }
            
            for pattern, key in mapping.items():
                if pattern in heading_text and not info[key]:
                    content = []
                    for sibling in tag.find_next_siblings():
                        if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                            break
                        content.append(sibling)
                    
                    if content:
                        temp_soup = BeautifulSoup("<div></div>", "html.parser")
                        for elem in content:
                            temp_soup.div.append(elem)
                        info[key] = self.extract_section_content(temp_soup.div)
                        break
        
        # Collect any remaining sections under other_information
        other_sections = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            heading_text = tag.get_text().strip()
            if heading_text and not any(key.replace("_", " ").lower() in heading_text.lower() for key in info.keys()):
                content = []
                for sibling in tag.find_next_siblings():
                    if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                        break
                    content.append(sibling)
                
                if content:
                    temp_soup = BeautifulSoup("<div></div>", "html.parser")
                    for elem in content:
                        temp_soup.div.append(elem)
                    section_content = self.extract_section_content(temp_soup.div)
                    if section_content:
                        other_sections.append(f"{heading_text}:\n{section_content}")
        
        if other_sections:
            info["other_information"] = "\n\n".join(other_sections)
        
        return info

    def scrape_product(self, product_name: str, product_url: str) -> Dict:
        """Scrape complete information for a single product"""
        logging.info(f"Scraping product: {product_name}")
        
        product_data = {
            "name": product_name,
            "url": product_url,
            "scraped_at": datetime.now().isoformat(),
            "prescribing_info": None
        }
        
        try:
            prescribing_url = self.get_prescribing_info_url(product_url)
            if prescribing_url:
                product_data["prescribing_info"] = self.parse_prescribing_info(prescribing_url)
            else:
                logging.warning(f"No prescribing information found for {product_name}")
        except Exception as e:
            logging.error(f"Error scraping {product_name}: {str(e)}")
        
        return product_data

    def save_product_data(self, product_name: str, data: Dict):
        """Save product data to a JSON file"""
        filename = f"{product_name.lower().replace(' ', '_')}.json"
        filepath = os.path.join(self.DATASETS_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved data for {product_name} to {filepath}")

    def run(self):
        """Run the complete scraping process"""
        products = self.get_all_product_urls()
        
        for product_name, url in products.items():
            try:
                product_data = self.scrape_product(product_name, url)
                self.save_product_data(product_name, product_data)
            except Exception as e:
                logging.error(f"Failed to process {product_name}: {str(e)}")

if __name__ == "__main__":
    scraper = MicroLabsScraper()
    scraper.run()
