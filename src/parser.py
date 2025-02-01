# /src/parser.py (c) 2025 RAGE

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import frontmatter
from markdown2 import Markdown
import yaml
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    """Parser for different document formats"""
    
    def __init__(self):
        self.markdown_converter = Markdown()
    
    def parse_file(self, file_path: Path) -> Tuple[str, Dict[str, Any], str]:
        """Parse file based on extension"""
        if file_path.suffix.lower() in ('.md', '.markdown'):
            content, metadata = self.parse_markdown(file_path)
            file_type = 'markdown'
        elif file_path.suffix.lower() == '.json':
            content, metadata = self.parse_json(file_path)
            file_type = 'json'
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        return content, metadata, file_type
    
    def parse_markdown(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse markdown files with frontmatter"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
                content = post.content
                metadata = post.metadata if post.metadata else {}
                plain_text = self.markdown_converter.convert(content)
                return plain_text, metadata
        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {e}")
            raise
    
    def parse_json(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = data.get('content', '')
                metadata = data.get('metadata', {})
                return content, metadata
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            raise
