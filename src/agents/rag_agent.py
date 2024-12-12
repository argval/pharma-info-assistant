import json
import os
import logging
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_agent.log')
    ]
)
logger = logging.getLogger(__name__)

class PharmaceuticalRAGAgent:
    def __init__(self, data_dir: str, model_name: str = "llama3.2"):
        """Initialize the RAG agent with the path to the pharmaceutical data directory."""
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
            
        self.data_dir = data_dir
        self.model_name = model_name
        self.query_cache = {}  # Cache for storing query results
        self.last_query_time = 0  # Track last query time for rate limiting
        self.min_query_interval = 1  # Minimum seconds between queries
        
        logger.info(f"Initializing RAG agent with model: {model_name}")
        
        try:
            # Use a simpler embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embeddings model initialized")
            
            self.documents = []
            self.vectorstore = None
            self.qa_chain = None
            
            # Initialize the system
            self._load_data()
            self._setup_vectorstore()
            self._setup_qa_chain()
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG agent components: {e}", exc_info=True)
            raise

    def _load_data(self):
        """Load and process all JSON files from the data directory."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            # Process each JSON file in the directory
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json') and not filename.startswith('.'):
                    file_path = os.path.join(self.data_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                        # Extract drug name from filename
                        drug_name = filename.replace('.json', '').replace('_', ' ').title()
                        
                        # Convert data into text format
                        product_text = f"Drug: {drug_name}\n\n"
                        
                        def process_dict(d, prefix=''):
                            text = ""
                            for key, value in d.items():
                                if isinstance(value, dict):
                                    text += f"{prefix}{key}:\n"
                                    text += process_dict(value, prefix + '  ')
                                elif isinstance(value, list):
                                    text += f"{prefix}{key}:\n"
                                    for item in value:
                                        if isinstance(item, dict):
                                            text += process_dict(item, prefix + '  ')
                                        else:
                                            text += f"{prefix}  - {item}\n"
                                else:
                                    text += f"{prefix}{key}: {value}\n"
                            return text
                        
                        product_text += process_dict(data)
                        
                        # Split the text into chunks
                        chunks = text_splitter.split_text(product_text)
                        self.documents.extend(chunks)
                        
                    except Exception as e:
                        logger.error(f"Error processing file {filename}: {e}")
                        continue
            
            logger.info(f"Loaded {len(self.documents)} document chunks from {self.data_dir}")
        except Exception as e:
            logger.error(f"Error loading data from directory {self.data_dir}: {e}")
            raise

    def _search_web(self, drug_name: str) -> str:
        """
        Search multiple medical websites for drug information when not found in local database.
        Returns consolidated information from reliable sources.
        """
        try:
            # List of trusted medical domains with their importance ranking
            TRUSTED_SOURCES = [
                {
                    'url': f"https://www.drugs.com/{drug_name.lower().replace(' ', '-')}.html",
                    'content_class': 'contentBox',
                    'name': 'Drugs.com',
                    'rank': 1.5
                },
                {
                    'url': f"https://www.webmd.com/drugs/2/drug-{drug_name.lower().replace(' ', '-')}/details",
                    'content_class': 'monograph-content',
                    'name': 'WebMD',
                    'rank': 1.3
                },
                {
                    'url': f"https://medlineplus.gov/druginfo/meds/a{drug_name.lower().replace(' ', '')}.html",
                    'content_class': 'section-body',
                    'name': 'MedlinePlus',
                    'rank': 1.4
                },
                {
                    'url': f"https://www.mayoclinic.org/drugs-supplements/{drug_name.lower().replace(' ', '-')}/description/drg-20",
                    'content_class': 'content',
                    'name': 'Mayo Clinic',
                    'rank': 1.5
                },
                {
                    'url': f"https://www.rxlist.com/search/rxl?q={drug_name.lower()}",
                    'content_class': 'drug-content',
                    'name': 'RxList',
                    'rank': 1.2
                }
            ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            
            drug_info = []
            for source in TRUSTED_SOURCES:
                try:
                    # Rate limiting
                    time.sleep(1)  # Be polite to servers
                    
                    response = requests.get(source['url'], headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                            element.decompose()
                        
                        # Try different content selectors
                        content = None
                        selectors = [
                            {'class': source['content_class']},
                            {'id': 'contentArea'},
                            {'class': 'drug-content'},
                            {'class': 'drug-info'},
                            {'role': 'main'},
                            {'class': re.compile(r'(content|article|main).*')}
                        ]
                        
                        for selector in selectors:
                            content = soup.find(['main', 'article', 'div'], selector)
                            if content:
                                break
                        
                        if content:
                            # Extract and clean text
                            paragraphs = content.find_all(['p', 'li', 'h2', 'h3'])
                            text_parts = []
                            
                            for p in paragraphs:
                                # Skip if paragraph contains unwanted content
                                if any(skip in p.get_text().lower() for skip in ['advertisement', 'subscribe', 'newsletter']):
                                    continue
                                    
                                text = p.get_text(strip=True)
                                if text and len(text) > 20:  # Skip very short snippets
                                    text_parts.append(text)
                            
                            if text_parts:
                                # Join paragraphs and clean text
                                text = ' '.join(text_parts)
                                text = re.sub(r'\s+', ' ', text)
                                text = text[:3000] + "..." if len(text) > 3000 else text
                                
                                # Store with source ranking
                                drug_info.append({
                                    'text': text,
                                    'source': source['name'],
                                    'rank': source['rank']
                                })
                                logger.info(f"Successfully extracted content from {source['name']}")
                
                except requests.RequestException as e:
                    logger.warning(f"Error fetching data from {source['name']}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing content from {source['name']}: {e}")
                    continue
            
            if drug_info:
                # Sort by source ranking
                drug_info.sort(key=lambda x: x['rank'], reverse=True)
                
                # Combine information from all sources
                combined_info = "\n\n".join(
                    f"Source: {info['source']} (Trusted Medical Source)\n{info['text']}"
                    for info in drug_info
                )
                
                logger.info(f"Successfully retrieved information for {drug_name} from {len(drug_info)} sources")
                return combined_info
            
            # If no information found, try a Google search as fallback
            try:
                search_url = f"https://www.google.com/search?q={drug_name}+drug+medical+information+site:nih.gov+OR+site:mayoclinic.org+OR+site:drugs.com"
                response = requests.get(search_url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract first few Google results
                results = []
                for result in soup.select('.g')[:3]:
                    link = result.find('a')
                    if link and 'href' in link.attrs:
                        url = link['href']
                        if url.startswith('http') and not url.startswith('https://google.com'):
                            results.append(url)
                
                if results:
                    # Try to fetch content from Google results
                    for url in results:
                        try:
                            time.sleep(1)  # Rate limiting
                            response = requests.get(url, headers=headers, timeout=10)
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.text, 'html.parser')
                                
                                # Remove unwanted elements
                                for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                                    element.decompose()
                                
                                # Extract main content
                                content = soup.find(['main', 'article']) or soup.find('div', class_=re.compile(r'(content|article)'))
                                if content:
                                    text = content.get_text(separator=' ', strip=True)
                                    text = re.sub(r'\s+', ' ', text)
                                    text = text[:3000] + "..." if len(text) > 3000 else text
                                    
                                    domain = urlparse(url).netloc
                                    return f"Source: {domain}\n{text}\n\nNote: This information was found through a general web search."
                        
                        except Exception as e:
                            logger.warning(f"Error fetching Google result {url}: {e}")
                            continue
            
            except Exception as e:
                logger.warning(f"Error in Google fallback search: {e}")
            
            logger.warning(f"No information found for {drug_name} from any source")
            return f"Could not find reliable information for {drug_name}. Please consult a healthcare professional."
            
        except Exception as e:
            logger.error(f"Error in web search for {drug_name}: {e}")
            return f"Error searching for {drug_name} information. Please try again later."

    def _setup_vectorstore(self):
        """Set up the FAISS vector store with the processed documents."""
        if not self.documents:
            logger.error("No documents loaded")
            raise ValueError("No documents loaded")
        
        try:
            self.vectorstore = FAISS.from_texts(
                texts=self.documents,
                embedding=self.embeddings
            )
            logger.info("Successfully created vector store")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def _setup_qa_chain(self):
        """Set up the retrieval QA chain."""
        try:
            # Initialize Ollama LLM
            llm = Ollama(model=self.model_name)

            # Create a custom prompt template
            prompt_template = """You are a knowledgeable pharmaceutical assistant. 
Use the following pieces of context to answer the question at the end. 
If the information is not found in the context, indicate that you'll search external sources.

Context: {context}

Question: {question}

Helpful Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )

            # Create the QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={
                    "prompt": PROMPT,
                }
            )
            logger.info("Successfully set up QA chain")
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            raise

    def query(self, query: str) -> str:
        """Process a query and return relevant information."""
        try:
            # Check if it's a symptom-based query
            symptom_keywords = ["pain", "ache", "fever", "cough", "headache", "nausea", "dizzy", "dizziness"]
            is_symptom_query = any(keyword in query.lower() for keyword in symptom_keywords)

            if is_symptom_query:
                prompt_template = """You are a pharmaceutical assistant. Given the following context and a user's symptom, 
                provide a clear, structured response about potential treatments. Include:
                1. Common over-the-counter medications
                2. Important precautions
                3. When to seek medical attention

                Context: {context}
                Question: {question}

                Response format:
                Recommended Medications:
                - [medication names and basic dosage]

                Precautions:
                - [important safety information]

                When to Seek Medical Help:
                - [warning signs]

                Note: This is general information only. Please consult a healthcare professional for personalized medical advice.
                """
            else:
                prompt_template = """You are a pharmaceutical assistant. Based on the following context, provide 
                detailed information about the medication or drug-related query. Include relevant safety information 
                and precautions.

                Context: {context}
                Question: {question}

                Please provide a clear, structured response including:
                - Main information
                - Usage and dosage
                - Important precautions
                - Potential side effects
                """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            chain_type_kwargs = {"prompt": PROMPT}
            
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_query_time < self.min_query_interval:
                time.sleep(self.min_query_interval)
            
            # Check cache first
            if query in self.query_cache:
                return self.query_cache[query]

            # Create QA chain with the custom prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=Ollama(model=self.model_name),
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True
            )

            # Get response
            response = qa_chain({"query": query})
            result = response['result']

            # Add source attribution if available
            if hasattr(response, 'source_documents') and response.source_documents:
                sources = set()
                for doc in response.source_documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.add(doc.metadata['source'])
                
                if sources:
                    result += "\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)

            # Add medical disclaimer
            result += "\n\nDisclaimer: This information is for general guidance only and should not replace professional medical advice."

            # Cache the result
            self.query_cache[query] = result
            self.last_query_time = time.time()

            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"I apologize, but I encountered an error while processing your query. Please try rephrasing your question or ask about a specific medication. Error: {str(e)}"

    def reset_conversation(self):
        """Reset the conversation history."""
        logger.info("Conversation reset called (no-op)")
        pass
