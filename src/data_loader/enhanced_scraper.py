import json
import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from tqdm import tqdm
import concurrent.futures
import validators
import random

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

@dataclass
class DrugSection:
    """Standardized structure for drug information sections"""
    content: str
    source_url: str
    last_updated: str
    confidence_score: float  # 0-1 score indicating content quality
    metadata: Dict

@dataclass
class DrugInformation:
    """Complete drug information structure"""
    name: str
    generic_name: Optional[str]
    brand_names: List[str]
    manufacturer: str
    ndc_codes: List[str]
    dosage_forms: List[str]
    indications_and_usage: DrugSection
    dosage_and_administration: DrugSection
    contraindications: DrugSection
    warnings_and_precautions: DrugSection
    adverse_reactions: DrugSection
    drug_interactions: DrugSection
    use_in_specific_populations: DrugSection
    clinical_pharmacology: DrugSection
    how_supplied: DrugSection
    additional_information: Dict[str, DrugSection]
    last_updated: str
    sources: List[str]
    metadata: Dict

class RateLimiter:
    """Manage rate limiting for web scraping"""
    def __init__(self, max_requests_per_minute=10):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = []

    def wait(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we've hit the max requests, wait
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
            time.sleep(max(0, sleep_time))
        
        # Record this request time
        self.request_times.append(current_time)

class ProgressTracker:
    """Track scraping progress and enable resumability"""
    def __init__(self, db_path: str = "scraper_progress.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    url TEXT PRIMARY KEY,
                    status TEXT,
                    attempts INTEGER DEFAULT 0,
                    last_attempt TEXT,
                    error_message TEXT
                )
            """)

    def mark_started(self, url: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO progress (url, status, attempts, last_attempt) VALUES (?, 'in_progress', 1, ?)",
                (url, datetime.now().isoformat())
            )

    def mark_completed(self, url: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE progress SET status = 'completed', last_attempt = ? WHERE url = ?",
                (datetime.now().isoformat(), url)
            )

    def mark_failed(self, url: str, error: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE progress 
                SET status = 'failed', 
                    attempts = attempts + 1,
                    last_attempt = ?,
                    error_message = ?
                WHERE url = ?
            """, (datetime.now().isoformat(), str(error), url))

    def get_pending_urls(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT url FROM progress WHERE status != 'completed' AND attempts < 3"
            )
            return [row[0] for row in cursor.fetchall()]

class EnhancedScraper:
    """Enhanced pharmaceutical data scraper with improved reliability and validation"""
    
    def __init__(self, output_dir: str = "./datasets/processed"):
        self.session = self._create_session()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = RateLimiter(max_requests_per_minute=5)  # More conservative rate limiting
        self.progress = ProgressTracker()
        self.sources = self._load_sources()

    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers and retry configuration"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PharmaBot/1.1; +http://example.com/bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        return session

    def _load_sources(self) -> Dict[str, Dict]:
        """Load pharmaceutical data sources configuration"""
        # In a real implementation, this would load from a configuration file
        return {
            "dailymed": {
                "base_url": "https://dailymed.nlm.nih.gov/dailymed/",
                "search_url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm",
                "priority": 1,
                "section_mappings": {
                    "indications and usage": "indications_and_usage",
                    "dosage and administration": "dosage_and_administration",
                    "contraindications": "contraindications",
                    "warnings and precautions": "warnings_and_precautions",
                    "adverse reactions": "adverse_reactions",
                    "drug interactions": "drug_interactions",
                    "use in specific populations": "use_in_specific_populations",
                    "clinical pharmacology": "clinical_pharmacology",
                    "how supplied": "how_supplied"
                }
            },
            "microlabs": {
                "base_url": "https://www.microlabsusa.com",
                "products_url": "https://www.microlabsusa.com/products/",
                "priority": 2,
                "section_mappings": {
                    "indications and usage": "indications_and_usage",
                    "dosage and administration": "dosage_and_administration",
                    "contraindications": "contraindications",
                    "warnings and precautions": "warnings_and_precautions",
                    "adverse reactions": "adverse_reactions",
                    "drug interactions": "drug_interactions",
                    "use in specific populations": "use_in_specific_populations",
                    "clinical pharmacology": "clinical_pharmacology",
                    "how supplied": "how_supplied"
                }
            }
            # Add more sources as needed
        }

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_fixed(2) + wait_exponential(multiplier=1, min=4, max=10)
    )
    def _fetch_page(self, url: str, timeout: int = 30) -> Optional[BeautifulSoup]:
        """Fetch a page with retry logic, rate limiting, and validation"""
        try:
            # Apply rate limiting
            self.rate_limiter.wait()

            # Validate URL
            if not validators.url(url):
                logger.warning(f"Invalid URL: {url}")
                return None

            # Fetch page with timeout
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            if not soup.find():  # Basic validation that we got HTML content
                logger.warning(f"Invalid HTML content received from {url}")
                return None
                
            return soup

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            # Add some randomized backoff to prevent overwhelming the server
            time.sleep(random.uniform(1, 5))
            raise

    def _extract_section_content(self, section_soup) -> Tuple[str, float]:
        """Extract content from a section with improved parsing and confidence scoring"""
        content = []
        confidence_score = 1.0
        
        # Process different content types
        for element in section_soup.find_all(['p', 'table', 'ul', 'ol', 'div']):
            if element.name == 'table':
                table_content = self._process_table(element)
                content.append(table_content)
            elif element.name in ['ul', 'ol']:
                list_content = self._process_list(element)
                content.append(list_content)
            else:
                text = element.get_text(strip=True)
                if text:
                    content.append(text)

        # Calculate confidence score based on content quality
        if not content:
            confidence_score = 0.0
        else:
            # Implement more sophisticated confidence scoring based on:
            # - Content length
            # - Presence of expected keywords
            # - Structure completeness
            # - Cross-references
            total_length = sum(len(c) for c in content)
            confidence_score *= min(1.0, total_length / 1000)  # Adjust based on expected length

        return "\n\n".join(content), confidence_score

    def _process_table(self, table_soup) -> str:
        """Process table content with improved formatting"""
        rows = []
        headers = []
        
        # Extract headers
        for th in table_soup.find_all('th'):
            headers.append(th.get_text(strip=True))
        
        if headers:
            rows.append(" | ".join(headers))
            rows.append("-" * len(" | ".join(headers)))
        
        # Extract data rows
        for tr in table_soup.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if cells and not all(cell in headers for cell in cells):
                rows.append(" | ".join(cells))
        
        return "\n".join(rows)

    def _process_list(self, list_soup) -> str:
        """Process list content with improved formatting"""
        items = []
        for item in list_soup.find_all('li'):
            text = item.get_text(strip=True)
            if text:
                items.append(f"- {text}")
        return "\n".join(items)

    def _validate_drug_info(self, drug_info: DrugInformation) -> bool:
        """Validate drug information completeness and quality"""
        required_sections = [
            'indications_and_usage',
            'dosage_and_administration',
            'contraindications',
            'warnings_and_precautions'
        ]
        
        # Check required sections have content
        for section in required_sections:
            section_data = getattr(drug_info, section)
            if not section_data.content or section_data.confidence_score < 0.3:
                logger.warning(f"Missing or low quality content in required section: {section}")
                return False
        
        # Validate basic drug information
        if not drug_info.name or not drug_info.manufacturer:
            logger.warning("Missing basic drug information")
            return False
            
        return True

    def scrape_drug(self, drug_name: str) -> Optional[DrugInformation]:
        """
        Scrape drug information from local dataset or multiple sources
        """
        # Check if drug info is already available locally
        local_file = os.path.join(self.output_dir, f"{drug_name.lower().replace(' ', '_')}.json")
        if os.path.exists(local_file):
            logger.info(f"Found local data for {drug_name}, loading...")
            try:
                with open(local_file, 'r') as f:
                    local_data = json.load(f)
                    return DrugInformation(
                        name=drug_name,
                        generic_name=local_data['_metadata']['generic_name'],
                        brand_names=local_data['_metadata']['brand_names'],
                        manufacturer=local_data['_metadata']['manufacturer'],
                        ndc_codes=local_data['_metadata']['ndc_codes'],
                        dosage_forms=local_data['_metadata']['dosage_forms'],
                        indications_and_usage=DrugSection(
                            content=local_data['indications_and_usage']['content'],
                            source_url=local_data['indications_and_usage']['source_url'],
                            last_updated=local_data['indications_and_usage']['last_updated'],
                            confidence_score=local_data['indications_and_usage']['confidence_score'],
                            metadata=local_data['indications_and_usage']['metadata']
                        ),
                        dosage_and_administration=DrugSection(
                            content=local_data['dosage_and_administration']['content'],
                            source_url=local_data['dosage_and_administration']['source_url'],
                            last_updated=local_data['dosage_and_administration']['last_updated'],
                            confidence_score=local_data['dosage_and_administration']['confidence_score'],
                            metadata=local_data['dosage_and_administration']['metadata']
                        ),
                        contraindications=DrugSection(
                            content=local_data['contraindications']['content'],
                            source_url=local_data['contraindications']['source_url'],
                            last_updated=local_data['contraindications']['last_updated'],
                            confidence_score=local_data['contraindications']['confidence_score'],
                            metadata=local_data['contraindications']['metadata']
                        ),
                        warnings_and_precautions=DrugSection(
                            content=local_data['warnings_and_precautions']['content'],
                            source_url=local_data['warnings_and_precautions']['source_url'],
                            last_updated=local_data['warnings_and_precautions']['last_updated'],
                            confidence_score=local_data['warnings_and_precautions']['confidence_score'],
                            metadata=local_data['warnings_and_precautions']['metadata']
                        ),
                        adverse_reactions=DrugSection(
                            content=local_data['adverse_reactions']['content'],
                            source_url=local_data['adverse_reactions']['source_url'],
                            last_updated=local_data['adverse_reactions']['last_updated'],
                            confidence_score=local_data['adverse_reactions']['confidence_score'],
                            metadata=local_data['adverse_reactions']['metadata']
                        ),
                        drug_interactions=DrugSection(
                            content=local_data['drug_interactions']['content'],
                            source_url=local_data['drug_interactions']['source_url'],
                            last_updated=local_data['drug_interactions']['last_updated'],
                            confidence_score=local_data['drug_interactions']['confidence_score'],
                            metadata=local_data['drug_interactions']['metadata']
                        ),
                        use_in_specific_populations=DrugSection(
                            content=local_data['use_in_specific_populations']['content'],
                            source_url=local_data['use_in_specific_populations']['source_url'],
                            last_updated=local_data['use_in_specific_populations']['last_updated'],
                            confidence_score=local_data['use_in_specific_populations']['confidence_score'],
                            metadata=local_data['use_in_specific_populations']['metadata']
                        ),
                        clinical_pharmacology=DrugSection(
                            content=local_data['clinical_pharmacology']['content'],
                            source_url=local_data['clinical_pharmacology']['source_url'],
                            last_updated=local_data['clinical_pharmacology']['last_updated'],
                            confidence_score=local_data['clinical_pharmacology']['confidence_score'],
                            metadata=local_data['clinical_pharmacology']['metadata']
                        ),
                        how_supplied=DrugSection(
                            content=local_data['how_supplied']['content'],
                            source_url=local_data['how_supplied']['source_url'],
                            last_updated=local_data['how_supplied']['last_updated'],
                            confidence_score=local_data['how_supplied']['confidence_score'],
                            metadata=local_data['how_supplied']['metadata']
                        ),
                        additional_information={},
                        last_updated=local_data['_metadata']['last_updated'],
                        sources=local_data['_metadata']['sources'],
                        metadata=local_data['_metadata']
                    )
            except Exception as e:
                logger.error(f"Failed to load local data for {drug_name}: {e}")

        # If not found locally, proceed with web scraping
        logger.info(f"Local data not found for {drug_name}, scraping from web...")
        combined_info = None
        for source_name, source_config in sorted(self.sources.items(), key=lambda x: x[1].get('priority', 999)):
            try:
                logger.info(f"Scraping from source: {source_name}")
                drug_info = self._scrape_from_source(drug_name, source_config)
                
                if drug_info:
                    if not combined_info:
                        combined_info = drug_info
                    else:
                        combined_info = self._merge_drug_info(combined_info, drug_info)
                        
            except Exception as e:
                logger.error(f"Error scraping {drug_name} from {source_name}: {str(e)}")
                continue
        
        if combined_info and self._validate_drug_info(combined_info):
            self._save_drug_info(combined_info)
            return combined_info
        
        return None

    def _scrape_from_source(self, drug_name: str, source_config: Dict) -> Optional[DrugInformation]:
        """Scrape drug information from a specific source"""
        # Implementation would vary based on source
        if 'dailymed' in source_config['base_url']:
            return self._scrape_from_dailymed(drug_name, source_config)
        elif 'microlabs' in source_config['base_url']:
            return self._scrape_from_microlabs(drug_name, source_config)
        else:
            logger.warning(f"Unsupported source: {source_config['base_url']}")
            return None

    def _merge_drug_info(self, primary: DrugInformation, secondary: DrugInformation) -> DrugInformation:
        """Merge drug information from multiple sources, preferring higher quality content"""
        merged = primary
        
        # Merge each section, keeping the higher quality content
        for field in DrugInformation.__dataclass_fields__:
            if field.endswith('_section'):
                primary_section = getattr(primary, field)
                secondary_section = getattr(secondary, field)
                
                if secondary_section.confidence_score > primary_section.confidence_score:
                    setattr(merged, field, secondary_section)
        
        # Update metadata and sources
        merged.sources = list(set(primary.sources + secondary.sources))
        merged.last_updated = max(primary.last_updated, secondary.last_updated)
        
        return merged

    def _save_drug_info(self, drug_info: DrugInformation):
        """Save drug information to JSON file with proper formatting"""
        filename = f"{drug_info.name.lower().replace(' ', '_')}.json"
        filepath = self.output_dir / filename
        
        # Convert to dictionary with proper formatting
        data = asdict(drug_info)
        
        # Add metadata
        data['_metadata'] = {
            'version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'schema_version': '2.0.0',
            'quality_score': self._calculate_quality_score(drug_info),
            'generic_name': drug_info.generic_name,
            'brand_names': drug_info.brand_names,
            'manufacturer': drug_info.manufacturer,
            'ndc_codes': drug_info.ndc_codes,
            'dosage_forms': drug_info.dosage_forms,
            'last_updated': drug_info.last_updated,
            'sources': drug_info.sources
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved drug information to {filepath}")

    def _calculate_quality_score(self, drug_info: DrugInformation) -> float:
        """Calculate overall quality score for the drug information"""
        scores = []
        weights = {
            'indications_and_usage': 1.0,
            'dosage_and_administration': 1.0,
            'contraindications': 0.8,
            'warnings_and_precautions': 0.8,
            'adverse_reactions': 0.6,
            'drug_interactions': 0.6,
            'clinical_pharmacology': 0.4
        }
        
        for field, weight in weights.items():
            section = getattr(drug_info, field)
            scores.append(section.confidence_score * weight)
        
        return sum(scores) / sum(weights.values())

    def _scrape_from_dailymed(self, drug_name: str, source_config: Dict) -> Optional[DrugInformation]:
        """Scrape drug information from DailyMed"""
        try:
            # Search for the drug
            search_url = source_config['search_url']
            params = {
                'searchterm': drug_name,
                'searchtype': 'name'
            }
            response = self.session.get(search_url, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the first result link
            result_link = soup.find('a', {'class': 'search-result'})
            if not result_link:
                logger.warning(f"No results found for {drug_name} on DailyMed")
                return None
                
            # Get the drug page
            drug_url = source_config['base_url'] + result_link['href']
            drug_soup = self._fetch_page(drug_url)
            
            if not drug_soup:
                return None
                
            # Extract sections
            sections = {}
            for section in drug_soup.find_all('div', {'class': 'Section'}):
                heading = section.find(['h1', 'h2', 'h3'])
                if not heading:
                    continue
                    
                heading_text = heading.get_text(strip=True).lower()
                content, confidence = self._extract_section_content(section)
                
                # Map section to standardized name
                for pattern, key in source_config['section_mappings'].items():
                    if pattern in heading_text:
                        sections[key] = DrugSection(
                            content=content,
                            source_url=drug_url,
                            last_updated=datetime.now().isoformat(),
                            confidence_score=confidence,
                            metadata={'source': 'dailymed'}
                        )
                        break
            
            # Create DrugInformation object
            return DrugInformation(
                name=drug_name,
                generic_name=self._extract_generic_name(drug_soup),
                brand_names=self._extract_brand_names(drug_soup),
                manufacturer=self._extract_manufacturer(drug_soup),
                ndc_codes=self._extract_ndc_codes(drug_soup),
                dosage_forms=self._extract_dosage_forms(drug_soup),
                indications_and_usage=sections.get('indications_and_usage', DrugSection('', '', '', 0.0, {})),
                dosage_and_administration=sections.get('dosage_and_administration', DrugSection('', '', '', 0.0, {})),
                contraindications=sections.get('contraindications', DrugSection('', '', '', 0.0, {})),
                warnings_and_precautions=sections.get('warnings_and_precautions', DrugSection('', '', '', 0.0, {})),
                adverse_reactions=sections.get('adverse_reactions', DrugSection('', '', '', 0.0, {})),
                drug_interactions=sections.get('drug_interactions', DrugSection('', '', '', 0.0, {})),
                use_in_specific_populations=sections.get('use_in_specific_populations', DrugSection('', '', '', 0.0, {})),
                clinical_pharmacology=sections.get('clinical_pharmacology', DrugSection('', '', '', 0.0, {})),
                how_supplied=sections.get('how_supplied', DrugSection('', '', '', 0.0, {})),
                additional_information={},
                last_updated=datetime.now().isoformat(),
                sources=['dailymed'],
                metadata={'source_url': drug_url}
            )
            
        except Exception as e:
            logger.error(f"Error scraping {drug_name} from DailyMed: {str(e)}")
            return None

    def _scrape_from_microlabs(self, drug_name: str, source_config: Dict) -> Optional[DrugInformation]:
        """Scrape drug information from MicroLabs"""
        try:
            # Search for the drug on the products page
            products_url = source_config['products_url']
            response = self.session.get(products_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find product link
            product_link = None
            for link in soup.find_all('a', href=True):
                if drug_name.lower() in link.get_text().strip().lower():
                    product_link = link
                    break
                    
            if not product_link:
                logger.warning(f"No results found for {drug_name} on MicroLabs")
                return None
                
            # Get the drug page
            drug_url = product_link['href']
            if not drug_url.startswith('http'):
                drug_url = source_config['base_url'] + drug_url
            drug_soup = self._fetch_page(drug_url)
            
            if not drug_soup:
                return None
                
            # Extract sections
            sections = {}
            for section in drug_soup.find_all(['div', 'section']):
                heading = section.find(['h1', 'h2', 'h3', 'h4'])
                if not heading:
                    continue
                    
                heading_text = heading.get_text(strip=True).lower()
                content, confidence = self._extract_section_content(section)
                
                # Map section to standardized name
                for pattern, key in source_config['section_mappings'].items():
                    if pattern in heading_text:
                        sections[key] = DrugSection(
                            content=content,
                            source_url=drug_url,
                            last_updated=datetime.now().isoformat(),
                            confidence_score=confidence,
                            metadata={'source': 'microlabs'}
                        )
                        break
            
            # Create DrugInformation object
            return DrugInformation(
                name=drug_name,
                generic_name=self._extract_generic_name(drug_soup),
                brand_names=self._extract_brand_names(drug_soup),
                manufacturer="MicroLabs",
                ndc_codes=self._extract_ndc_codes(drug_soup),
                dosage_forms=self._extract_dosage_forms(drug_soup),
                indications_and_usage=sections.get('indications_and_usage', DrugSection('', '', '', 0.0, {})),
                dosage_and_administration=sections.get('dosage_and_administration', DrugSection('', '', '', 0.0, {})),
                contraindications=sections.get('contraindications', DrugSection('', '', '', 0.0, {})),
                warnings_and_precautions=sections.get('warnings_and_precautions', DrugSection('', '', '', 0.0, {})),
                adverse_reactions=sections.get('adverse_reactions', DrugSection('', '', '', 0.0, {})),
                drug_interactions=sections.get('drug_interactions', DrugSection('', '', '', 0.0, {})),
                use_in_specific_populations=sections.get('use_in_specific_populations', DrugSection('', '', '', 0.0, {})),
                clinical_pharmacology=sections.get('clinical_pharmacology', DrugSection('', '', '', 0.0, {})),
                how_supplied=sections.get('how_supplied', DrugSection('', '', '', 0.0, {})),
                additional_information={},
                last_updated=datetime.now().isoformat(),
                sources=['microlabs'],
                metadata={'source_url': drug_url}
            )
            
        except Exception as e:
            logger.error(f"Error scraping {drug_name} from MicroLabs: {str(e)}")
            return None

    def _extract_generic_name(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract generic name from drug page"""
        generic_patterns = ['generic name:', 'generic:', 'active ingredient:']
        for pattern in generic_patterns:
            element = soup.find(lambda tag: tag.name in ['p', 'div', 'span'] and 
                              pattern in tag.get_text().strip().lower())
            if element:
                return element.get_text().strip().split(':')[-1].strip()
        return None

    def _extract_brand_names(self, soup: BeautifulSoup) -> List[str]:
        """Extract brand names from drug page"""
        brand_patterns = ['brand name:', 'brand names:', 'trade name:']
        brands = []
        for pattern in brand_patterns:
            element = soup.find(lambda tag: tag.name in ['p', 'div', 'span'] and 
                              pattern in tag.get_text().strip().lower())
            if element:
                text = element.get_text().strip().split(':')[-1]
                brands.extend([b.strip() for b in text.split(',')])
        return list(set(brands)) if brands else []

    def _extract_manufacturer(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract manufacturer from drug page"""
        mfg_patterns = ['manufacturer:', 'manufactured by:', 'distributed by:']
        for pattern in mfg_patterns:
            element = soup.find(lambda tag: tag.name in ['p', 'div', 'span'] and 
                              pattern in tag.get_text().strip().lower())
            if element:
                return element.get_text().strip().split(':')[-1].strip()
        return None

    def _extract_ndc_codes(self, soup: BeautifulSoup) -> List[str]:
        """Extract NDC codes from drug page"""
        ndc_patterns = ['ndc:', 'ndc code:', 'ndc number:']
        codes = []
        for pattern in ndc_patterns:
            elements = soup.find_all(lambda tag: tag.name in ['p', 'div', 'span'] and 
                                   pattern in tag.get_text().strip().lower())
            for element in elements:
                text = element.get_text().strip().split(':')[-1]
                codes.extend([c.strip() for c in text.split(',')])
        return list(set(codes)) if codes else []

    def _extract_dosage_forms(self, soup: BeautifulSoup) -> List[str]:
        """Extract dosage forms from drug page"""
        form_patterns = ['dosage form:', 'form:', 'available forms:']
        forms = []
        for pattern in form_patterns:
            elements = soup.find_all(lambda tag: tag.name in ['p', 'div', 'span'] and 
                                   pattern in tag.get_text().strip().lower())
            for element in elements:
                text = element.get_text().strip().split(':')[-1]
                forms.extend([f.strip() for f in text.split(',')])
        return list(set(forms)) if forms else []

    def run(self, drug_list: List[str], max_workers: int = 4):
        """Run the scraper for multiple drugs concurrently with more robust error handling"""
        logger.info(f"Starting scraping for {len(drug_list)} drugs")
        
        # Track overall progress and results
        scraped_drugs = []
        failed_drugs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all drug scraping tasks
            future_to_drug = {
                executor.submit(self._safe_scrape_drug, drug): drug 
                for drug in drug_list
            }
            
            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_drug),
                total=len(drug_list),
                desc="Scraping Drugs"
            ):
                drug = future_to_drug[future]
                try:
                    result = future.result()
                    if result:
                        scraped_drugs.append(drug)
                        logger.info(f"Successfully scraped {drug}")
                    else:
                        failed_drugs.append(drug)
                        logger.warning(f"Failed to scrape {drug}")
                except Exception as e:
                    failed_drugs.append(drug)
                    logger.error(f"Error scraping {drug}: {str(e)}")

        # Log summary
        logger.info(f"Scraping complete. Scraped: {len(scraped_drugs)}, Failed: {len(failed_drugs)}")
        
        if failed_drugs:
            logger.warning(f"Failed to scrape the following drugs: {', '.join(failed_drugs)}")

    def _safe_scrape_drug(self, drug_name: str) -> Optional[Dict]:
        """Safely scrape a single drug with comprehensive error handling"""
        try:
            combined_info = None
            for source_name, source_config in sorted(self.sources.items(), key=lambda x: x[1].get('priority', 999)):
                try:
                    logger.info(f"Attempting to scrape {drug_name} from {source_name}")
                    drug_info = self._scrape_from_source(drug_name, source_config)
                    
                    if drug_info:
                        if not combined_info:
                            combined_info = drug_info
                        else:
                            combined_info = self._merge_drug_info(combined_info, drug_info)
                            
                except Exception as e:
                    logger.error(f"Error scraping {drug_name} from {source_name}: {str(e)}")
                    continue
            
            if combined_info and self._validate_drug_info(combined_info):
                self._save_drug_info(combined_info)
                return combined_info
            
            return None

        except Exception as e:
            logger.error(f"Unexpected error in safe drug scrape for {drug_name}: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    scraper = EnhancedScraper()
    drug_list = [
        "Amoxicillin",
        "Lisinopril",
        "Metformin",
        "Omeprazole",
        "Simvastatin"
    ]
    scraper.run(drug_list)
