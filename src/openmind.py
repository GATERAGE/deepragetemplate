# src/openmind.py (c) 2025 Gregory L. Magnusson MIT license

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv, set_key, find_dotenv, get_key
from pathlib import Path
import logging
import secrets

logger = logging.getLogger('rage.openmind')

class OpenMind:
    """Central configuration and resource management for RAGE"""
    
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenMind, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.env_file = '.env'
        
        # Define base paths
        self.base_path = Path(__file__).parent.parent
        
        # System resources paths
        self.system_resources = {
            'prompt': str(self.base_path / 'prompt.txt'),
            'system_prompt': str(self.base_path / 'prompt.txt')
        }
        
        # Memory structure
        self.memory_structure = {
            'root': str(self.base_path / 'memory'),
            'folders': {
                'logs': str(self.base_path / 'memory/logs'),
                'api': str(self.base_path / 'memory/api'),
                'models': str(self.base_path / 'memory/models'),
                'cache': str(self.base_path / 'memory/cache'),
                'index': str(self.base_path / 'memory/index'),
                'responses': str(self.base_path / 'memory/responses')
            }
        }
        
        # Resource cache
        self._resource_cache = {}
        
        # Session tracking
        self.current_session = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "api_operations": [],
            "resources_loaded": set(),
            "errors": []
        }
        
        try:
            self._initialize_system()
        except Exception as e:
            logger.error(f"Critical initialization error: {e}")
            raise

    def _initialize_system(self):
        """Initialize complete system"""
        try:
            # Create directory structure
            self._initialize_memory()
            
            # Verify resources exist
            self._verify_resources()
            
            # Ensure .env file exists with proper permissions
            self._initialize_env_file()
            
            logger.info("System initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

    def _initialize_env_file(self):
        """Initialize .env file with proper permissions"""
        try:
            env_path = Path(self.env_file)
            if not env_path.exists():
                env_path.touch(mode=0o600)  # Create with restricted permissions
            else:
                os.chmod(self.env_file, 0o600)  # Update permissions if file exists
            
            # Ensure file has required variables (empty if not set)
            required_vars = ['OPENAI_API_KEY', 'GROQ_API_KEY', 'TOGETHER_API_KEY']
            for var in required_vars:
                if not get_key(self.env_file, var):
                    set_key(self.env_file, var, "")
                    
        except Exception as e:
            logger.error(f"Error initializing .env file: {e}")
            raise

    def _initialize_memory(self):
        """Initialize memory structure"""
        try:
            created_folders = []
            for folder in self.memory_structure['folders'].values():
                os.makedirs(folder, exist_ok=True)
                created_folders.append(folder)
            
            logger.info(f"Memory structure initialized: {created_folders}")
            
        except Exception as e:
            logger.error(f"Failed to create memory structure: {e}")
            raise

    def _verify_resources(self):
        """Verify system resources exist"""
        try:
            missing_resources = []
            for resource_name, resource_path in self.system_resources.items():
                if not os.path.exists(resource_path):
                    missing_resources.append(resource_name)
            
            if missing_resources:
                logger.warning(f"Missing resources: {missing_resources}")
            
        except Exception as e:
            logger.error(f"Resource verification failed: {e}")
            raise

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key directly from .env file"""
        try:
            service_upper = service.upper()
            env_var = f"{service_upper}_API_KEY"
            
            # Force reload .env file
            load_dotenv(self.env_file, override=True)
            
            # Get key directly from environment
            key = os.getenv(env_var)
            
            if not key:
                logger.warning(f"API key not found for {service}")
                return None
                
            return key
            
        except Exception as e:
            logger.error(f"Error getting API key: {e}")
            return None

    def save_api_key(self, service: str, key: str) -> bool:
        """Save or update API key"""
        try:
            service_upper = service.upper()
            env_var = f"{service_upper}_API_KEY"
            
            # Validate key format
            if not self._validate_api_key(service, key):
                raise ValueError(f"Invalid API key format for {service}")
            
            # Save to env file
            set_key(self.env_file, env_var, key)
            
            # Track operation
            self._log_api_operation(service, "key_updated")
            
            logger.info(f"API key updated for {service}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving API key for {service}: {e}")
            return False

    def update_api_key(self, service: str, key: str) -> bool:
        """Update existing API key"""
        return self.save_api_key(service, key)

    def remove_api_key(self, service: str) -> bool:
        """Remove API key"""
        try:
            service_upper = service.upper()
            env_var = f"{service_upper}_API_KEY"
            
            # Clear key in env file
            set_key(self.env_file, env_var, "")
            
            # Track operation
            self._log_api_operation(service, "key_removed")
            
            logger.info(f"API key removed for {service}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing API key for {service}: {e}")
            return False

    def _validate_api_key(self, service: str, key: str) -> bool:
        """Validate API key format"""
        try:
            if not key:
                return False
                
            # Service-specific validation
            if service.lower() == 'openai':
                return key.startswith('sk-') and len(key) > 20
            elif service.lower() == 'together':
                return len(key) > 20
            elif service.lower() == 'groq':
                return len(key) > 20
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return False

    def _log_api_operation(self, service: str, action: str):
        """Log API key operations"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "service": service,
                "action": action,
                "session_id": self.current_session["session_id"]
            }
            
            # Add to session tracking
            self.current_session["api_operations"].append(log_entry)
            
            # Save to log file
            log_file = Path(self.memory_structure['folders']['api']) / 'api_operations.json'
            self._append_to_json_log(log_file, log_entry)
            
        except Exception as e:
            logger.error(f"Error logging API operation: {e}")

    def _append_to_json_log(self, filepath: Path, entry: Dict):
        """Append entry to JSON log file"""
        try:
            if filepath.exists():
                with open(filepath, 'r+') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    data.append(entry)
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
            else:
                with open(filepath, 'w') as f:
                    json.dump([entry], f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error appending to log file: {e}")

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt"""
        return self._load_resource('/prompt/prompt.txt')

    def _load_resource(self, resource_name: str) -> Optional[str]:
        """Load a system resource with caching"""
        try:
            # Check cache first
            if resource_name in self._resource_cache:
                return self._resource_cache[resource_name]
            
            # Load from file
            resource_path = self.system_resources.get(resource_name)
            if not resource_path:
                logger.error(f"Unknown resource: {resource_name}")
                return None
                
            if not os.path.exists(resource_path):
                logger.error(f"Resource file not found: {resource_path}")
                return None
                
            with open(resource_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                self._resource_cache[resource_name] = content
                self.current_session["resources_loaded"].add(resource_name)
                return content
                
        except Exception as e:
            logger.error(f"Error loading resource: {e}")
            self.current_session["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "resource": resource_name
            })
            return None

    def get_session_status(self) -> Dict:
        """Get current session status"""
        try:
            # Get current API key status without exposing keys
            api_status = {
                service: bool(self.get_api_key(service))
                for service in ['openai', 'groq', 'together']
            }
            
            return {
                "session_id": self.current_session["session_id"],
                "start_time": self.current_session["start_time"],
                "duration": (datetime.now() - datetime.fromisoformat(
                    self.current_session["start_time"]
                )).total_seconds(),
                "api_operations": len(self.current_session["api_operations"]),
                "resources_loaded": list(self.current_session["resources_loaded"]),
                "error_count": len(self.current_session["errors"]),
                "api_keys_configured": api_status
            }
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return {}

    def verify_api_keys(self) -> Dict[str, bool]:
        """Verify all API keys"""
        return {
            service: bool(self.get_api_key(service))
            for service in ['openai', 'groq', 'together']
        }
