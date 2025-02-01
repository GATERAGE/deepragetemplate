# mdmbed.py (c) 2025 web3dguy (Modified for RAGE)

import json
import re
import os
import logging
from tqdm import tqdm
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer

class RAGEmbedder:
    """Document processing and embedding for RAGE"""
    
    def __init__(self, 
                 knowledge_dir: str = "./knowledge",
                 embedding_model: str = "nomic-embed-text",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.knowledge_dir = Path(knowledge_dir)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger('rage.embedder')
        
        # Initialize directories
        self.setup_directories()
        
        # Initialize embeddings
        self.setup_embeddings()
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.knowledge_dir,
            self.knowledge_dir / "markdown",
            self.knowledge_dir / "embeddings",
            self.knowledge_dir / "vectors"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_embeddings(self):
        """Initialize embedding models"""
        try:
            # Try Ollama first
            self.embeddings = OllamaEmbeddings(
                base_url="http://localhost:11434",
                model=self.embedding_model
            )
        except Exception as e:
            self.logger.warning(f"Falling back to SentenceTransformers: {e}")
            self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    
    def load_documents(self, content: str, metadata: Optional[Dict] = None) -> List[Document]:
        """Load and process documents"""
        try:
            if not metadata:
                metadata = {}
            
            # Create document
            doc = Document(page_content=content, metadata=metadata)
            
            # Split document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            return text_splitter.split_documents([doc])
            
        except Exception as e:
            self.logger.error(f"Error loading document: {e}")
            return []
    
    def process_and_embed(self, 
                         content: str,
                         metadata: Optional[Dict] = None,
                         collection_name: str = "rage_docs") -> bool:
        """Process and embed documents"""
        try:
            # Load and split documents
            documents = self.load_documents(content, metadata)
            
            if not documents:
                return False
            
            # Initialize vector store
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.knowledge_dir / "vectors")
            )
            
            # Generate UUIDs
            uuids = [str(uuid4()) for _ in range(len(documents))]
            
            # Add documents to vector store
            vector_store.add_documents(documents=documents, ids=uuids)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing and embedding document: {e}")
            return False
    
    def search_similar(self, 
                      query: str,
                      collection_name: str = "rage_docs",
                      k: int = 3) -> List[Document]:
        """Search for similar documents"""
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.knowledge_dir / "vectors")
            )
            
            return vector_store.similarity_search(query, k=k)
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def get_collection_info(self, collection_name: str = "rage_docs") -> Dict:
        """Get information about the collection"""
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.knowledge_dir / "vectors")
            )
            
            return {
                "name": collection_name,
                "count": vector_store._collection.count()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {"name": collection_name, "count": 0}

# Example usage
if __name__ == "__main__":
    # Initialize embedder
    embedder = RAGEEmbedder()
    
    # Example document
    content = """
    RAGE (Retrieval Augmented Generation Environment) is a system that combines
    document retrieval with language model generation. It supports multiple models
    and provides efficient document management.
    """
    
    metadata = {
        "title": "RAGE Overview",
        "source": "documentation"
    }
    
    # Process and embed
    if embedder.process_and_embed(content, metadata):
        print("Document processed successfully")
        
        # Search example
        results = embedder.search_similar("What is RAGE?")
        for doc in results:
            print(f"\nFound document:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
    
    # Get collection info
    info = embedder.get_collection_info()
    print(f"\nCollection info: {info}")
