# deeprage.py (c) 2025 RAGE
# document embeddings specific to DeekSeek RAGE as standalone local RAGE example file
from typing import List, Dict, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
from dataclasses import dataclass, asdict

@dataclass
class Document:
    """Document structure for RAGE"""
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None
    doc_id: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class RAGEProcessor:
    """Enhanced document processor for RAGE with DeepSeek"""
    
    def __init__(self, 
                 ollama_base_url: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text",
                 vector_dim: int = 384,
                 knowledge_dir: str = "./knowledge"):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim
        self.knowledge_dir = Path(knowledge_dir)
        
        # Setup logging
        self.logger = logging.getLogger('rage.processor')
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(vector_dim)
        self.documents: List[Document] = []
        
        # Fallback embedding model
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create directories
        self._setup_directories()
        
        # Load existing index if available
        self._load_existing_index()
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.knowledge_dir,
            self.knowledge_dir / "vectors",
            self.knowledge_dir / "documents",
            self.knowledge_dir / "cache"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_existing_index(self):
        """Load existing index and documents if available"""
        try:
            index_path = self.knowledge_dir / "vectors/index.faiss"
            docs_path = self.knowledge_dir / "documents/documents.json"
            
            if index_path.exists() and docs_path.exists():
                self.index = faiss.read_index(str(index_path))
                with open(docs_path, 'r') as f:
                    docs_data = json.load(f)
                    self.documents = [Document(**doc) for doc in docs_data]
                self.logger.info(f"Loaded {len(self.documents)} documents from existing index")
        except Exception as e:
            self.logger.error(f"Error loading existing index: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings with fallback"""
        try:
            # Try Ollama first
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=10
            )
            
            if response.status_code == 200:
                embedding = response.json()['embedding']
                return np.array(embedding, dtype=np.float32)
            
            raise Exception(f"Ollama error: {response.text}")
            
        except Exception as e:
            self.logger.warning(f"Falling back to sentence-transformers: {e}")
            return self.st_model.encode(text, show_progress_bar=False)
    
    def add_document(self, 
                    content: str, 
                    metadata: Optional[Dict] = None,
                    batch_mode: bool = False) -> Optional[str]:
        """Add document to index"""
        try:
            # Get embedding
            embedding = self.get_embedding(content)
            
            # Create document
            doc = Document(
                content=content,
                metadata=metadata or {},
                embedding=embedding.tolist(),
                doc_id=f"doc_{len(self.documents)}"
            )
            
            # Add to index
            self.index.add(np.array([embedding]))
            self.documents.append(doc)
            
            # Save if not in batch mode
            if not batch_mode:
                self.save_index()
            
            return doc.doc_id
            
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return None
    
    def add_documents_batch(self, 
                          documents: List[Dict[str, Union[str, Dict]]]) -> List[str]:
        """Add multiple documents efficiently"""
        doc_ids = []
        try:
            for doc in tqdm(documents, desc="Adding documents"):
                doc_id = self.add_document(
                    content=doc['content'],
                    metadata=doc.get('metadata', {}),
                    batch_mode=True
                )
                doc_ids.append(doc_id)
            
            # Save after batch processing
            self.save_index()
            return doc_ids
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return doc_ids
    
    def search(self, 
              query: str, 
              k: int = 3,
              threshold: float = 0.8) -> List[Dict]:
        """Search with similarity threshold"""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Search index
            D, I = self.index.search(
                np.array([query_embedding]), 
                min(k * 2, len(self.documents))  # Get more results for filtering
            )
            
            # Filter and format results
            results = []
            for i, idx in enumerate(I[0]):
                if idx < len(self.documents) and D[0][i] < threshold:
                    doc = self.documents[idx]
                    results.append({
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'score': float(D[0][i]),
                        'doc_id': doc.doc_id
                    })
            
            # Return top k after filtering
            return results[:k]
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def save_index(self):
        """Save index and documents"""
        try:
            # Save FAISS index
            index_path = self.knowledge_dir / "vectors/index.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save documents
            docs_path = self.knowledge_dir / "documents/documents.json"
            with open(docs_path, 'w') as f:
                json.dump([asdict(doc) for doc in self.documents], f, indent=2)
            
            self.logger.info(f"Saved {len(self.documents)} documents to index")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "vector_dimension": self.vector_dim,
            "embedding_model": self.embedding_model
        }

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = RAGEProcessor()
    
    # Add some documents
    docs = [
        {
            "content": "DeepSeek is a powerful language model available through Ollama.",
            "metadata": {"type": "description", "model": "deepseek"}
        },
        {
            "content": "RAGE uses vector similarity search for context retrieval.",
            "metadata": {"type": "technical", "component": "search"}
        }
    ]
    
    # Batch add
    doc_ids = processor.add_documents_batch(docs)
    
    # Search
    results = processor.search("What is DeepSeek?")
    
    print("\nSearch Results:")
    for result in results:
        print(f"\nScore: {result['score']:.4f}")
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")
    
    # Print stats
    print("\nIndex Stats:")
    print(processor.get_stats())
