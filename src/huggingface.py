# /src/huggingface.py (c) 2025 RAGE

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class HuggingFaceHandler:
    def __init__(self, 
                 model_name: str = "deepseek/deepseek-llm-7b-chat",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.error = None
        
        try:
            # Initialize main model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(embedding_model)
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error initializing HuggingFace models: {e}")
    
    def generate_response(self, prompt: str, 
                         max_length: int = 500,
                         temperature: float = 0.7) -> Optional[str]:
        try:
            outputs = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1
            )
            return outputs[0]["generated_text"]
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error generating response: {e}")
            return None
    
    def get_embeddings(self, text: str) -> Optional[torch.Tensor]:
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def get_last_error(self) -> Optional[str]:
        return self.error
