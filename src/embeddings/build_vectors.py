"""
Script to build and test vector store for pharmaceutical data.
"""

import os
from pathlib import Path
from src.embeddings.vectorstore import PharmaVectorStore

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    processed_dir = base_dir / 'datasets' / 'processed'
    embeddings_dir = base_dir / 'datasets' / 'embeddings'
    
    # Initialize vector store
    vector_store = PharmaVectorStore(
        processed_data_dir=str(processed_dir),
        embeddings_dir=str(embeddings_dir),
        allow_dangerous_deserialization=True
    )
    
    # Build vector store
    print("Building vector store...")
    vector_store.build_vector_store()
    
    # Test similarity search
    print("\nTesting similarity search...")
    test_queries = [
        "What are the drug interactions for Timolol?",
        "What are the common side effects of ophthalmic solutions?",
        "What is the recommended dosage for treating glaucoma?",
        "List contraindications for beta blockers",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Product: {doc.metadata['product_name']}")
            print(f"Section: {doc.metadata['section']}")
            print(f"Content: {doc.page_content[:200]}...")

if __name__ == '__main__':
    main()
