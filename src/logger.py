# src/logger.py (c) 2025 Gregory L. Magnusson MIT license

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict, Optional
import traceback

class RAGEFormatter(logging.Formatter):
    """Custom formatter for RAGE logging"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.structured_keys = ['session_id', 'model', 'provider', 'error']
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with additional structured data"""
        # Basic message formatting
        message = super().format(record)
        
        # Add structured data if available
        if hasattr(record, 'structured_data'):
            structured_data = record.structured_data
            structured_str = ' '.join(
                f"{k}={v!r}" for k, v in structured_data.items()
                if k in self.structured_keys and v is not None
            )
            if structured_str:
                message = f"{message} [{structured_str}]"
        
        return message

class JSONFormatter(logging.Formatter):
    """JSON formatter for file logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add structured data if available
        if hasattr(record, 'structured_data'):
            log_data['structured_data'] = record.structured_data
        
        return json.dumps(log_data)

def setup_logger(name: str, 
                log_dir: Optional[Path] = None,
                level: int = logging.INFO) -> logging.Logger:
    """Setup logger with both console and file handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(RAGEFormatter())
    logger.addHandler(console_handler)
    
    # File handlers if directory is provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Regular log file
        file_handler = logging.FileHandler(
            log_dir / f"{name}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(RAGEFormatter())
        logger.addHandler(file_handler)
        
        # JSON log file
        json_handler = logging.FileHandler(
            log_dir / f"{name}_json.log",
            encoding='utf-8'
        )
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)
        
        # Error log file
        error_handler = logging.FileHandler(
            log_dir / f"{name}_error.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        logger.addHandler(error_handler)
    
    return logger

class StructuredLogger:
    """Logger wrapper for structured logging"""
    
    def __init__(self, name: str, session_id: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.session_id = session_id
        self.context = {}
    
    def add_context(self, **kwargs):
        """Add context to all subsequent log messages"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear current context"""
        self.context = {}
    
    def _log(self, level: int, msg: str, 
             structured_data: Optional[Dict[str, Any]] = None, 
             **kwargs):
        """Internal logging method"""
        if structured_data is None:
            structured_data = {}
        
        # Add session_id and context
        if self.session_id:
            structured_data['session_id'] = self.session_id
        structured_data.update(self.context)
        
        # Add extra to structured_data
        extra = kwargs.pop('extra', {})
        if isinstance(extra, dict):
            structured_data.update(extra)
        
        # Create log record
        extra = {'structured_data': structured_data}
        self.logger.log(level, msg, extra=extra, **kwargs)
    
    def debug(self, msg: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.DEBUG, msg, structured_data, **kwargs)
    
    def info(self, msg: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.INFO, msg, structured_data, **kwargs)
    
    def warning(self, msg: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.WARNING, msg, structured_data, **kwargs)
    
    def error(self, msg: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.ERROR, msg, structured_data, **kwargs)
    
    def critical(self, msg: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.CRITICAL, msg, structured_data, **kwargs)

def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger"""
    log_dir = Path("logs")
    setup_logger(name, log_dir)
    return StructuredLogger(name)

# Example usage
if __name__ == "__main__":
    logger = get_logger("rage")
    
    # Add context
    logger.add_context(
        model="deepseek-coder:6.7b",
        provider="ollama"
    )
    
    # Log with structured data
    logger.info(
        "Processing request",
        structured_data={
            "query": "What is RAGE?",
            "response_time": 0.5
        }
    )
    
    # Log error with exception
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error(
            "Error processing request",
            structured_data={
                "error_type": type(e).__name__,
                "error_details": str(e)
            },
            exc_info=True
        )
