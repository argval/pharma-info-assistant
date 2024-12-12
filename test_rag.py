from src.agents.rag_agent import PharmaRAGAgent

def main():
    # Initialize the RAG agent with llama2:3.2
    agent = PharmaRAGAgent()
    
    # Test query
    query = "What are the common side effects of aspirin?"
    print(f"\nQuery: {query}")
    
    # Get response
    response = agent.query(query)
    
    # Print response
    print("\nAnswer:", response["answer"])
    print("\nSources:")
    for source in response["sources"]:
        print(f"- {source}")

if __name__ == "__main__":
    main()
