"""
Script to run data preprocessing on scraped pharmaceutical data.
"""

import os
from pathlib import Path
from preprocessor import PharmaceuticalDataPreprocessor
import json

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / 'datasets' / 'microlabs_usa'
    output_dir = base_dir / 'datasets' / 'processed'
    
    # Create preprocessor
    preprocessor = PharmaceuticalDataPreprocessor(input_dir, output_dir)
    
    # Process all files
    print("Starting data preprocessing...")
    preprocessor.process_all_files()
    
    # Generate and save statistics
    print("\nGenerating statistics...")
    stats = preprocessor.generate_statistics()
    
    # Save statistics
    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nStatistics Summary:")
    print(f"Total files processed: {stats['total_files']}")
    print("\nSection presence:")
    for section, count in stats['sections_present'].items():
        print(f"- {section}: {count}/{stats['total_files']} files")
    
    print("\nAverage section lengths (characters):")
    for section, avg_len in stats['avg_section_length'].items():
        print(f"- {section}: {avg_len}")
    
    print("\nEmpty sections:")
    for section, count in stats['empty_sections'].items():
        print(f"- {section}: {count} files")

if __name__ == '__main__':
    main()
