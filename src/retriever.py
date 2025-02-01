# /src/retriever.py (c) 2025 RAGE

from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import faiss
from huggingface_hub import HuggingFaceEmbeddings
from tqdm import tqdm
from .document import Document
from .store import DocumentStore
import logging

logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """Enhanced retrieval system"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 knowledge_dir: str = "./knowledge"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.knowledge_dir = Path(knowledge_dir)
        self.document_store = DocumentStore(knowledge_dir)
        self.index = None
        self.doc_ids = []
    
    def create_index(self, documents: List[Document]):
        """Create FAISS index from documents"""
        embeddings = []
        self.doc_ids = []
        
        logger.info("Generating embeddings for documents...")
        for doc in tqdm(documents):
            embedding = self.embedding_model.embed(doc.content)
            embeddings.append(embedding)
            self.doc_ids.append(doc.doc_id)
            doc.embedding = embedding
        
        embeddings_array = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        
        # Save index
        faiss.write_index(self.index, str(self.knowledge_dir / "index/index.faiss"))
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant documents"""
        query_embedding = self.embedding_model.embed(query)
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        return [
            self.document_store.documents[self.doc_ids[i]]
            for i in indices[0]
        ]
