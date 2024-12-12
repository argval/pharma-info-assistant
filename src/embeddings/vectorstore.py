"""
Vector store module for pharmaceutical data RAG system.
Uses LangChain with FAISS for efficient similarity search.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PharmaVectorStore:
    """Vector store for pharmaceutical data using FAISS."""
    
    def __init__(self, processed_data_dir: str, embeddings_dir: str, allow_dangerous_deserialization: bool = False):
        """
        Initialize the vector store.
        
        Args:
            processed_data_dir: Directory containing processed JSON files
            embeddings_dir: Directory to save FAISS index
            allow_dangerous_deserialization: Whether to allow pickle deserialization
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.allow_dangerous_deserialization = allow_dangerous_deserialization
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        self.vector_store = None
    
    def _prepare_document(self, data: Dict[str, Any]) -> List[Document]:
        """Prepare document chunks from pharmaceutical data."""
        documents = []
        
        # Extract product information
        product_name = data['name']
        
        # Process each section of prescribing information
        if 'prescribing_info' in data:
            for section, content in data['prescribing_info'].items():
                if content:
                    # Create a formatted text with section context
                    text = f"Product: {product_name}\nSection: {section}\nContent: {content}"
                    
                    # Create metadata for better retrieval
                    metadata = {
                        'product_name': product_name,
                        'section': section,
                        'url': data.get('url', ''),
                        'source_type': 'prescribing_info'
                    }
                    
                    # Split text into chunks
                    chunks = self.text_splitter.split_text(text)
                    
                    # Create documents from chunks
                    for chunk in chunks:
                        doc = Document(
                            page_content=chunk,
                            metadata=metadata
                        )
                        documents.append(doc)
        
        return documents
    
    def build_vector_store(self) -> None:
        """Build the vector store from processed pharmaceutical data."""
        all_documents = []
        processed_files = 0
        skipped_files = 0
        
        # Process each JSON file
        for file_path in self.processed_data_dir.glob('*.json'):
            logging.info(f"Processing {file_path.name} for vector store")
            
            try:
                # Load and process the file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Validate data structure
                if not data.get('prescribing_info') or not isinstance(data['prescribing_info'], dict):
                    logging.warning(f"Skipping {file_path.name}: Missing or invalid prescribing information")
                    skipped_files += 1
                    continue
                
                if not any(data['prescribing_info'].values()):
                    logging.warning(f"Skipping {file_path.name}: Empty prescribing information")
                    skipped_files += 1
                    continue
                
                # Prepare documents from the data
                documents = self._prepare_document(data)
                if documents:
                    all_documents.extend(documents)
                    processed_files += 1
                    logging.info(f"Successfully processed {file_path.name} with {len(documents)} documents")
                else:
                    logging.warning(f"No documents generated for {file_path.name}")
                    skipped_files += 1
                
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {str(e)}")
                skipped_files += 1
        
        if not all_documents:
            raise ValueError("No valid documents found to build vector store")
        
        logging.info(f"Creating vector store from {len(all_documents)} documents")
        logging.info(f"Successfully processed {processed_files} files")
        logging.info(f"Skipped {skipped_files} files")
        
        # Create FAISS index
        self.vector_store = FAISS.from_documents(
            documents=all_documents,
            embedding=self.embeddings
        )
        
        # Save the index
        self.save_vector_store()
        
        logging.info("Vector store built and saved successfully")
    
    def save_vector_store(self) -> None:
        """Save the FAISS index to disk."""
        if self.vector_store:
            save_path = self.embeddings_dir / "pharma_index"
            self.vector_store.save_local(str(save_path))
            logging.info(f"Vector store saved to {save_path}")
    
    def load_vector_store(self) -> Optional[FAISS]:
        """Load the FAISS index from disk."""
        load_path = self.embeddings_dir / "pharma_index"
        if load_path.exists():
            self.vector_store = FAISS.load_local(
                str(load_path),
                self.embeddings,
                allow_dangerous_deserialization=self.allow_dangerous_deserialization
            )
            logging.info(f"Vector store loaded from {load_path}")
            return self.vector_store
        return None
    
    def get_retriever(self, k: int = 4) -> BaseRetriever:
        """Get a retriever for similarity search."""
        if not self.vector_store:
            self.load_vector_store()
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            self.load_vector_store()
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        
        return self.vector_store.similarity_search(query, k=k)
