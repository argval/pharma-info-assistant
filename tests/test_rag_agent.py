import unittest
from pathlib import Path
from src.agents.rag_agent import PharmaRAGAgent
from src.embeddings.vectorstore import PharmaVectorStore

class TestPharmaRAGAgent(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path("datasets/processed")
        self.vector_store = PharmaVectorStore()
        self.agent = PharmaRAGAgent()

    def test_query_processing(self):
        """Test basic query processing functionality"""
        query = "What are the side effects of Product X?"
        response = self.agent.query(query)
        self.assertIsInstance(response, dict)
        self.assertIn("answer", response)
        self.assertIn("sources", response)

    def test_source_tracking(self):
        """Test source tracking in responses"""
        query = "Tell me about Product Y's dosage information"
        response = self.agent.query(query)
        self.assertIsInstance(response, dict)
        self.assertIn("sources", response)
        self.assertIsInstance(response["sources"], list)

    def test_model_configuration(self):
        """Test model configuration options"""
        # Test with different model
        agent_mistral = PharmaRAGAgent(model_name="mistral")
        query = "What are the indications for Product Z?"
        response = agent_mistral.query(query)
        self.assertIsInstance(response, dict)
        self.assertIn("answer", response)

if __name__ == '__main__':
    unittest.main()
