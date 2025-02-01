# models.py (c) 2025 RAGE
from typing import Optional, Dict, Any
import requests
import logging
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger('rage.models')

class BaseHandler:
    """Base class for model handlers"""
    def __init__(self):
        self.error = None
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        raise NotImplementedError
    
    def get_last_error(self) -> Optional[str]:
        return self.error

class OllamaHandler(BaseHandler):
    """Handler for Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__()
        self.base_url = base_url
        self.model = None
    
    def select_model(self, model_name: str) -> bool:
        """Select Ollama model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            )
            response.raise_for_status()
            self.model = model_name
            return True
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error selecting model: {e}")
            return False
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using Ollama"""
        if not self.model:
            self.error = "No model selected"
            return "Error: No model selected"
        
        try:
            # Combine context and prompt if context is provided
            full_prompt = f"""
            Context: {context}
            
            Question: {prompt}
            
            Answer:""" if context else prompt
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            return response.json()["response"]
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

class HuggingFaceHandler(BaseHandler):
    """Handler for HuggingFace models"""
    
    def __init__(self, model_name: str = "deepseek/deepseek-llm-7b-chat"):
        super().__init__()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error initializing HuggingFace model: {e}")
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            full_prompt = f"""
            Context: {context}
            
            Question: {prompt}
            
            Answer:""" if context else prompt
            
            outputs = self.generator(
                full_prompt,
                max_length=500,
                temperature=0.7,
                num_return_sequences=1
            )
            return outputs[0]["generated_text"]
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

class GPT4Handler(BaseHandler):
    """Handler for OpenAI GPT-4"""
    
    def __init__(self, api_key: str):
        super().__init__()
        try:
            import openai
            openai.api_key = api_key
            self.client = openai.OpenAI()
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error initializing OpenAI: {e}")
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            messages = []
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Context: {context}"
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

class GroqHandler(BaseHandler):
    """Handler for Groq"""
    
    def __init__(self, api_key: str):
        super().__init__()
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error initializing Groq: {e}")
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            messages = []
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Context: {context}"
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768"
            )
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

class TogetherHandler(BaseHandler):
    """Handler for Together AI"""
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://api.together.xyz/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            messages = []
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Context: {context}"
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
