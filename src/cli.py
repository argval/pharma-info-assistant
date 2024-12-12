"""
Command-line interface for the Pharmaceutical Knowledge Assistant.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents.rag_agent import PharmaceuticalRAGAgent

def main():
    parser = argparse.ArgumentParser(description='Pharmaceutical Information Assistant')
    parser.add_argument('--data-dir', type=str, 
                       default=str(project_root / 'datasets' / 'microlabs_usa'),
                       help='Path to the pharmaceutical data directory')
    parser.add_argument('--model', type=str, default='llama3.2',
                       help='Ollama model to use (default: llama3.2)')
    args = parser.parse_args()

    # Initialize the RAG agent
    try:
        agent = PharmaceuticalRAGAgent(
            data_dir=args.data_dir,
            model_name=args.model
        )
        print("\nPharmaceutical Assistant initialized successfully!")
        print("This system uses both local Microlabs USA data and web search when needed.")
        print("\nYou can ask questions like:")
        print("- What are the side effects of Amoxicillin?")
        print("- What is the recommended dosage for Metformin?")
        print("- Tell me about drug interactions with Atorvastatin")
        print("\nCommands:")
        print("- Type 'quit' or 'exit' to end the session")
        print("- Type 'reset' to start a new conversation")
        
        while True:
            question = input("\nYou: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if question.lower() == 'reset':
                agent.reset_conversation()
                print("Conversation reset!")
                continue
            
            if not question:
                continue
            
            try:
                answer = agent.query(question)
                print("\nAssistant:", answer)
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again with a different question.")
    
    except Exception as e:
        print(f"Error initializing the assistant: {str(e)}")

if __name__ == "__main__":
    main()
