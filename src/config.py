# /src/config.py (c) 2025 rage

from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # Model settings
    MODEL_NAME: str = "deepseek-r1:1.5b"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Directory settings
    BASE_DIR: Path = Path("./knowledge")
    MARKDOWN_DIR: Path = BASE_DIR / "markdown"
    JSON_DIR: Path = BASE_DIR / "json"
    INDEX_DIR: Path = BASE_DIR / "index"
    CACHE_DIR: Path = BASE_DIR / "cache"
    RESPONSE_DIR: Path = BASE_DIR / "responses"
    
    # Generation settings
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500
    
    # Retrieval settings
    TOP_K: int = 3
    
    # File extensions
    MARKDOWN_EXTENSIONS: tuple = ('.md', '.markdown')
    JSON_EXTENSIONS: tuple = ('.json',)
