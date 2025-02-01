# /src/store.py (c) 2025 RAGE

from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
from datetime import datetime
from .document import Document
from .parser import DocumentParser
import logging

logger = logging.getLogger(__name__)

class DocumentStore:
    """Document storage and management system"""
    
    def __init__(self, base_dir: str = "./knowledge"):
        self.base_dir = Path(base_dir)
        self.documents: Dict[str, Document] = {}
        self.parser = DocumentParser()
        self.setup_directory_structure()
    
    def setup_directory_structure(self):
        """Create necessary directories"""
        directories = [
            self.base_dir,
            self.base_dir / "markdown",
            self.base_dir / "json",
            self.base_dir / "index",
            self.base_dir / "cache",
            self.base_dir / "responses"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def add_document(self, 
                    content: str, 
                    metadata: Dict[str, Any],
                    file_type: str,
                    original_file: Optional[str] = None) -> str:
        """Add document to store"""
        doc_id = f"doc_{len(self.documents)}_{datetime.now().timestamp()}"
        self.documents[doc_id] = Document(
            content=content,
            metadata=metadata,
            file_type=file_type,
            doc_id=doc_id,
            original_file=original_file
        )
        return doc_id
    
    def save_document(self, doc_id: str, output_format: Optional[str] = None):
        """Save document in specified format"""
        doc = self.documents[doc_id]
        output_format = output_format or doc.file_type
        
        if output_format == 'markdown':
            self._save_as_markdown(doc)
        elif output_format == 'json':
            self._save_as_json(doc)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _save_as_markdown(self, doc: Document):
        """Save as markdown with frontmatter"""
        output_path = self.base_dir / "markdown" / f"{doc.doc_id}.md"
        content = f"""---
{yaml.dump(doc.metadata)}
---

{doc.content}
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_as_json(self, doc: Document):
        """Save as JSON"""
        output_path = self.base_dir / "json" / f"{doc.doc_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc.to_dict(), f, indent=2)
