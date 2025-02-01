# /src/document.py (c) 2025 rage

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class Document:
    """Document class for storing content and metadata"""
    content: str
    metadata: Dict[str, Any]
    file_type: str
    embedding: Optional[np.ndarray] = None
    doc_id: Optional[str] = None
    original_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "file_type": self.file_type,
            "doc_id": self.doc_id,
            "original_file": str(self.original_file) if self.original_file else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary"""
        return cls(**data)
