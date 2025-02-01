# /src/utils.py (c) 2025 RAGE

import logging
from pathlib import Path
from typing import Union

def setup_logging(log_file: Union[str, Path] = None):
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def validate_directory(directory: Union[str, Path]) -> Path:
    """Validate and create directory if needed"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
