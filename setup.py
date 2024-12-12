from setuptools import setup, find_packages

setup(
    name="pharma-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4>=4.12.2",
        "requests>=2.31.0",
        "pandas>=2.1.4",
        "streamlit>=1.29.0",
        "faiss-cpu>=1.7.4",
        "langchain>=0.0.350",
        "langchain-community>=0.0.6",
        "sentence-transformers>=2.2.2",
        "torch>=2.1.2",
        "tqdm>=4.66.1",
        "python-dotenv>=1.0.0",
        "langchain-core>=0.1.0",
        "langchain-text-splitters>=0.0.1"
    ],
)
