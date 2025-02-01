# knowledge.py (c) 2025 RAGE
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import yaml
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from logger import get_logger

logger = get_logger('knowledge')

class Document:
    def __init__(self, content: str, metadata: Dict[str, Any], doc_id: str):
        self.content = content
        self.metadata = metadata
        self.doc_id = doc_id
        self.embedding = None

class KnowledgeBase:
    def __init__(self, base_dir: str = "./knowledge"):
        self.base_dir = Path(base_dir)
        self.documents: Dict[str, Document] = {}
        self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.index = None
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.base_dir,
            self.base_dir / "markdown",
            self.base_dir / "json",
            self.base_dir / "index",
            self.base_dir / "responses"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def add_document(self, content: str, metadata: Dict[str, Any], file_type: str) -> str:
        """Add document to knowledge base"""
        doc_id = f"doc_{len(self.documents)}_{datetime.now().timestamp()}"
        self.documents[doc_id] = Document(content, metadata, doc_id)
        self._update_index()
        return doc_id
    
    def _update_index(self):
        """Update or create FAISS index"""
        if not self.documents:
            return
        
        embeddings = []
        for doc in self.documents.values():
            embedding = self.embedding_model.encode(doc.content)
            embeddings.append(embedding)
            doc.embedding = embedding
        
        embeddings_array = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
        self.index.add(embeddings_array)
    
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context for query"""
        if not self.index or not self.documents:
            return ""
        
        query_embedding = self.embedding_model.encode(query)
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        relevant_docs = [
            list(self.documents.values())[i]
            for i in indices[0]
        ]
        
        return "\n\n".join([doc.content for doc in relevant_docs])
    
    def save_response(self, query: str, response: Dict[str, Any], format: str = 'markdown'):
        """Save response to file"""
        timestamp = datetime.now().isoformat()
        output_dir = self.base_dir / "responses"
        
        if format == 'markdown':
            output_path = output_dir / f"response_{timestamp}.md"
            content = f"""---
timestamp: {timestamp}
query: {query}
---

# Response

{response['response']}

## Sources Used

{yaml.dump(response['sources'])}
"""
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        elif format == 'json':
            output_path = output_dir / f"response_{timestamp}.json"
            data = {
                "query": query,
                "response": response,
                "timestamp": timestamp
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
