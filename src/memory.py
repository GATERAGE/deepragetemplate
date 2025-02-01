# memory.py (c) 2025 Gregory L. Magnusson MIT license

import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict, field
import hashlib
from logger import get_logger

@dataclass
class DialogEntry:
    """Structure for dialogue entries"""
    query: str
    response: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict = field(default_factory=dict)
    provider: str = None
    model: str = None
    sources: List[Dict] = field(default_factory=list)
    
    def to_json(self) -> Dict:
        """Convert entry to JSON-serializable dict"""
        return {
            'query': self.query,
            'response': self.response,
            'timestamp': self.timestamp,
            'context': self.context,
            'provider': self.provider,
            'model': self.model,
            'sources': self.sources
        }

@dataclass
class MemoryEntry:
    """Structure for memory entries"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    entry_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if self.entry_id is None:
            self.entry_id = hashlib.md5(
                f"{self.timestamp}{self.content}".encode()
            ).hexdigest()
    
    def to_json(self) -> Dict:
        """Convert entry to JSON-serializable dict"""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding,
            'entry_id': self.entry_id,
            'timestamp': self.timestamp
        }

class MemoryManager:
    """Memory management system for RAGE"""
    
    def __init__(self):
        self.logger = get_logger('memory')
        
        # Define memory structure
        self.base_dir = Path('./memory')
        self.memory_structure = {
            'conversations': self.base_dir / 'conversations',
            'knowledge': self.base_dir / 'knowledge',
            'embeddings': self.base_dir / 'embeddings',
            'cache': self.base_dir / 'cache'
        }
        
        # Initialize system
        self._initialize_memory_system()
        
        # Create session file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_session_file()

    def _initialize_memory_system(self):
        """Initialize memory system and create directories"""
        try:
            # Create all directories
            for directory in self.memory_structure.values():
                directory.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Memory system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory system: {e}")
            raise

    def _create_session_file(self):
        """Create session tracking file"""
        session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "entries": []
        }
        
        session_file = self.memory_structure['conversations'] / f"session_{self.session_id}.json"
        self._write_json(session_file, session_data)

    def _write_json(self, filepath: Path, data: Dict) -> bool:
        """Write data to JSON file with error handling"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write JSON file {filepath}: {e}")
            return False

    def _read_json(self, filepath: Path) -> Optional[Dict]:
        """Read JSON file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read JSON file {filepath}: {e}")
            return None

    def store_conversation(self, entry: DialogEntry) -> bool:
        """Store conversation entry as JSON"""
        try:
            # Generate unique ID for the entry
            entry_id = hashlib.md5(
                f"{entry.timestamp}{entry.query}".encode()
            ).hexdigest()
            
            # Prepare entry data
            entry_data = entry.to_json()
            entry_data.update({
                'entry_id': entry_id,
                'session_id': self.session_id
            })
            
            # Save conversation entry
            conv_file = self.memory_structure['conversations'] / f"conv_{entry_id}.json"
            if self._write_json(conv_file, entry_data):
                # Update session file
                session_file = self.memory_structure['conversations'] / f"session_{self.session_id}.json"
                session_data = self._read_json(session_file)
                if session_data:
                    session_data['entries'].append(entry_id)
                    self._write_json(session_file, session_data)
                
                self.logger.info(f"Stored conversation entry: {entry_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}")
            return False

    def store_knowledge(self, entry: MemoryEntry) -> bool:
        """Store knowledge entry as JSON"""
        try:
            # Save knowledge entry
            knowledge_file = self.memory_structure['knowledge'] / f"knowledge_{entry.entry_id}.json"
            knowledge_data = entry.to_json()
            
            # Store embedding separately if present
            if entry.embedding is not None:
                embedding_file = self.memory_structure['embeddings'] / f"embedding_{entry.entry_id}.json"
                embedding_data = {
                    'entry_id': entry.entry_id,
                    'embedding': entry.embedding
                }
                self._write_json(embedding_file, embedding_data)
                knowledge_data['has_embedding'] = True
            
            if self._write_json(knowledge_file, knowledge_data):
                self.logger.info(f"Stored knowledge entry: {entry.entry_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to store knowledge: {e}")
            return False

    def get_conversation_history(self, 
                               session_id: Optional[str] = None,
                               limit: int = 100) -> List[Dict]:
        """Retrieve conversation history from JSON files"""
        try:
            conversations = []
            session_id = session_id or self.session_id
            
            # Get session file
            session_file = self.memory_structure['conversations'] / f"session_{session_id}.json"
            session_data = self._read_json(session_file)
            
            if session_data and 'entries' in session_data:
                for entry_id in session_data['entries'][-limit:]:
                    conv_file = self.memory_structure['conversations'] / f"conv_{entry_id}.json"
                    if conv_data := self._read_json(conv_file):
                        conversations.append(conv_data)
            
            return conversations
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve conversation history: {e}")
            return []

    def clear_cache(self) -> bool:
        """Clear cache directory"""
        try:
            cache_dir = self.memory_structure['cache']
            for file in cache_dir.glob("*.json"):
                file.unlink()
            self.logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

# Global instance
memory_manager = MemoryManager()

# Convenience functions
def store_conversation(entry: DialogEntry) -> bool:
    """Store conversation entry"""
    return memory_manager.store_conversation(entry)

def store_knowledge(entry: MemoryEntry) -> bool:
    """Store knowledge entry"""
    return memory_manager.store_knowledge(entry)

def get_conversation_history(
    session_id: Optional[str] = None,
    limit: int = 100
) -> List[Dict]:
    """Get conversation history"""
    return memory_manager.get_conversation_history(session_id, limit)

def clear_cache() -> bool:
    """Clear cache"""
    return memory_manager.clear_cache()

# Example usage
if __name__ == "__main__":
    # Test conversation storage
    dialog = DialogEntry(
        query="What is RAGE?",
        response="RAGE is a Retrieval Augmented Generation Environment...",
        provider="ollama",
        model="deepseek-r1:1.5b"
    )
    store_conversation(dialog)
    
    # Test knowledge storage
    knowledge = MemoryEntry(
        content="RAGE system documentation...",
        metadata={"type": "documentation", "version": "1.0"},
        embedding=[0.1, 0.2, 0.3]
    )
    store_knowledge(knowledge)
    
    # Test retrieval
    history = get_conversation_history(limit=5)
    print(f"Retrieved {len(history)} conversations")
