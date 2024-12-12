"""
Data preprocessing module for cleaning and structuring pharmaceutical data.
"""

import json
import os
import re
import logging
from typing import Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PharmaceuticalDataPreprocessor:
    """Preprocesses pharmaceutical data for the RAG system."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Directory containing raw JSON files
            output_dir: Directory to save processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML artifacts
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up common artifacts
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        # Remove citation markers
        text = re.sub(r'\(\d+\)', '', text)
        
        return text.strip()
    
    def extract_sections(self, prescribing_info: Dict[str, str]) -> Dict[str, str]:
        """Extract and clean prescribing information sections."""
        cleaned_info = {}
        
        for section, content in prescribing_info.items():
            # Clean the section content
            cleaned_content = self.clean_text(content)
            
            # Skip empty sections
            if cleaned_content:
                # Convert section name to a standardized format
                section_name = section.lower().replace(' ', '_')
                cleaned_info[section_name] = cleaned_content
                
        return cleaned_info
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize the filename to be filesystem-friendly."""
        # Remove special characters and spaces
        name = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with underscores
        name = re.sub(r'\s+', '_', name)
        # Convert to lowercase
        return name.lower()

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single pharmaceutical data file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract basic information - fallback to filename if name not in data
            name = data.get('name')
            if not name:
                # Convert filename to product name
                name = file_path.stem
                name = name.replace('_', ' ').title()
            
            processed_data = {
                'name': name,
                'url': data.get('url', ''),
                'scraped_at': data.get('scraped_at', '')
            }
            
            # Process prescribing information
            if 'prescribing_info' in data and data['prescribing_info']:
                processed_data['prescribing_info'] = self.extract_sections(data['prescribing_info'])
            else:
                # Create empty prescribing info structure
                processed_data['prescribing_info'] = {}
            
            # Save with sanitized filename
            output_path = self.output_dir / f"{self.sanitize_filename(name)}.json"
            with open(output_path, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def process_all_files(self) -> None:
        """Process all pharmaceutical data files in the input directory."""
        processed_count = 0
        error_count = 0
        
        # Process each JSON file
        for file_path in self.input_dir.glob('*.json'):
            logging.info(f"Processing {file_path.name}")
            
            if self.process_file(file_path):
                processed_count += 1
            else:
                error_count += 1
        
        logging.info(f"Processing complete. Processed {processed_count} files. Errors: {error_count}")
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics about the processed data."""
        stats = {
            'total_files': 0,
            'sections_present': {},
            'avg_section_length': {},
            'empty_sections': {}
        }
        
        # Analyze all processed files
        for file_path in self.output_dir.glob('*.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            stats['total_files'] += 1
            
            # Analyze prescribing information sections
            if 'prescribing_info' in data:
                for section, content in data['prescribing_info'].items():
                    # Track section presence
                    stats['sections_present'][section] = stats['sections_present'].get(section, 0) + 1
                    
                    # Track section lengths
                    if content:
                        current_len = stats['avg_section_length'].get(section, [0, 0])
                        stats['avg_section_length'][section] = [
                            current_len[0] + len(content),
                            current_len[1] + 1
                        ]
                    else:
                        stats['empty_sections'][section] = stats['empty_sections'].get(section, 0) + 1
        
        # Calculate averages
        for section, (total_len, count) in stats['avg_section_length'].items():
            stats['avg_section_length'][section] = round(total_len / count, 2)
        
        return stats
