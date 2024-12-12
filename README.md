# Pharmaceutical Information Assistant

A sophisticated pharmaceutical information retrieval system that combines local dataset RAG (Retrieval Augmented Generation) with web-based information extraction.

## Features

- 🔍 Smart drug information retrieval from local and web sources
- 💊 Comprehensive pharmaceutical database
- 🌐 Web scraping from trusted medical sources
- 💬 Interactive chat interface using Streamlit
- 🤖 LLM-powered responses with Ollama
- 🔒 Source verification and ranking
- ⚡ Efficient caching mechanism

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is installed and running with llama3.2 model:
```bash
ollama pull llama2
```

4. Run the Streamlit app:
```bash
streamlit run src/app.py
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── app.py                 # Streamlit interface
│   └── agents/
│       ├── __init__.py
│       ├── rag_agent.py       # RAG implementation
│       └── internet_query.py  # Web scraping module
├── datasets/
│   ├── processed/            # Processed drug data
│   └── embeddings/           # Vector embeddings
└── tests/                    # Test files
```

## Dependencies

- streamlit
- langchain
- langchain-community
- sentence-transformers
- beautifulsoup4
- requests
- numpy
- faiss-cpu
- ollama
- tf-keras

## Features

### Local RAG System
- Efficient drug information retrieval
- Vector similarity search using FAISS
- Semantic understanding of queries

### Web Information Retrieval
- Trusted medical source scraping
- Source reliability ranking
- Content extraction and verification

### User Interface
- Clean, intuitive chat interface
- Real-time responses
- Conversation history
- Clear medical disclaimers

## Safety and Disclaimers

This tool is for informational purposes only. Always consult healthcare professionals for medical advice. The system includes:
- Clear medical disclaimers
- Source attribution
- Verification of information
- Warning signs for seeking medical attention

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical information sources: NIH, Mayo Clinic, WebMD, etc.
- LangChain for RAG implementation
- Ollama for local LLM support
