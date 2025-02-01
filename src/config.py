# src/config.py (c) 2025 Gregory L. Magnusson MIT license

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class ModelSettings:
    """Model-specific settings"""
    name: str
    developer: str
    tokens: int
    cost: str
    capabilities: List[str]

class ModelConfig:
    """Model configuration and provider settings"""
    
    def __init__(self):
        self.providers = {
            'openai': {
                'gpt-4': ModelSettings(
                    name='GPT-4',
                    developer='OpenAI',
                    tokens=8192,
                    cost='$0.03/1K tokens',
                    capabilities=['chat', 'rag', 'analysis', 'medical']
                ),
                'gpt-4-turbo': ModelSettings(
                    name='GPT-4 Turbo',
                    developer='OpenAI',
                    tokens=128000,
                    cost='$0.01/1K tokens',
                    capabilities=['chat', 'rag', 'analysis', 'medical']
                ),
                'gpt-3.5-turbo': ModelSettings(
                    name='GPT-3.5 Turbo',
                    developer='OpenAI',
                    tokens=16385,
                    cost='$0.002/1K tokens',
                    capabilities=['chat', 'rag']
                )
            },
            'groq': {
                'mixtral-8x7b-32768': ModelSettings(
                    name='Mixtral-8x7B',
                    developer='Mistral/Groq',
                    tokens=32768,
                    cost='$0.0002/1K tokens',
                    capabilities=['chat', 'rag', 'fast-inference']
                ),
                'llama2-70b-4096': ModelSettings(
                    name='LLaMA2 70B',
                    developer='Meta/Groq',
                    tokens=4096,
                    cost='$0.0004/1K tokens',
                    capabilities=['chat', 'rag']
                )
            },
            'together': {
                'mistralai/Mixtral-8x7B-Instruct-v0.1': ModelSettings(
                    name='Mixtral-8x7B-Instruct',
                    developer='Mistral',
                    tokens=32768,
                    cost='$0.0004/1K tokens',
                    capabilities=['chat', 'rag']
                ),
                'meta-llama/Llama-2-70b-chat-hf': ModelSettings(
                    name='Llama-2-70B',
                    developer='Meta',
                    tokens=4096,
                    cost='$0.0009/1K tokens',
                    capabilities=['chat', 'rag']
                )
            },
            'ollama': {
                'deepseek-coder:6.7b': ModelSettings(
                    name='DeepSeek Coder 6.7B',
                    developer='DeepSeek',
                    tokens=8192,
                    cost='$0.00/1K tokens',
                    capabilities=['chat', 'rag', 'code']
                ),
                'deepseek:67b': ModelSettings(
                    name='DeepSeek 67B',
                    developer='DeepSeek',
                    tokens=8192,
                    cost='$0.00/1K tokens',
                    capabilities=['chat', 'rag']
                ),
                'mixtral': ModelSettings(
                    name='Mixtral 8x7B',
                    developer='Mistral',
                    tokens=32768,
                    cost='$0.00/1K tokens',
                    capabilities=['chat', 'rag']
                ),
                'llama2': ModelSettings(
                    name='LLaMA 2',
                    developer='Meta',
                    tokens=4096,
                    cost='$0.00/1K tokens',
                    capabilities=['chat', 'rag']
                ),
                'codellama': ModelSettings(
                    name='Code LLaMA',
                    developer='Meta',
                    tokens=4096,
                    cost='$0.00/1K tokens',
                    capabilities=['chat', 'rag', 'code']
                ),
                'neural-chat': ModelSettings(
                    name='Neural Chat',
                    developer='Intel',
                    tokens=4096,
                    cost='$0.00/1K tokens',
                    capabilities=['chat', 'rag']
                )
            },
            'huggingface': {
                'deepseek/deepseek-llm-7b-chat': ModelSettings(
                    name='DeepSeek Chat 7B',
                    developer='DeepSeek',
                    tokens=8192,
                    cost='$0.00/1K tokens',
                    capabilities=['chat', 'rag']
                ),
                'mistralai/Mixtral-8x7B-Instruct-v0.1': ModelSettings(
                    name='Mixtral-8x7B-Instruct',
                    developer='Mistral',
                    tokens=32768,
                    cost='$0.00/1K tokens',
                    capabilities=['chat', 'rag']
                )
            }
        }

    def get_model_info(self, provider: str, model_id: str) -> Optional[ModelSettings]:
        """Get model information"""
        return self.providers.get(provider, {}).get(model_id)
    
    def get_provider_models(self, provider: str) -> Dict[str, ModelSettings]:
        """Get all models for a provider"""
        return self.providers.get(provider, {})
    
    def get_all_providers(self) -> List[str]:
        """Get list of all providers"""
        return list(self.providers.keys())
    
    def get_models_by_capability(self, capability: str) -> Dict[str, List[str]]:
        """Get models that have a specific capability"""
        models_with_capability = {}
        for provider, models in self.providers.items():
            capable_models = [
                model_id for model_id, settings in models.items()
                if capability in settings.capabilities
            ]
            if capable_models:
                models_with_capability[provider] = capable_models
        return models_with_capability

@dataclass
class Config:
    """Main configuration settings"""
    
    # Model settings
    DEFAULT_MODEL: str = "deepseek-coder:6.7b"
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
    SIMILARITY_THRESHOLD: float = 0.8
    
    # File extensions
    MARKDOWN_EXTENSIONS: tuple = ('.md', '.markdown')
    JSON_EXTENSIONS: tuple = ('.json',)
    
    # API settings
    API_TIMEOUT: int = 30
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: int = 1
    
    # Cache settings
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    # Embedding settings
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_LENGTH: int = 512
    
    # UI settings
    MAX_HISTORY_LENGTH: int = 100
    STREAM_OUTPUT: bool = True
    
    def __post_init__(self):
        """Ensure directories exist after initialization"""
        for directory in [
            self.BASE_DIR,
            self.MARKDOWN_DIR,
            self.JSON_DIR,
            self.INDEX_DIR,
            self.CACHE_DIR,
            self.RESPONSE_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)

# Create instances
config = Config()
model_config = ModelConfig()

def get_config() -> Config:
    """Get configuration instance"""
    return config

def get_model_config() -> ModelConfig:
    """Get model configuration instance"""
    return model_config
