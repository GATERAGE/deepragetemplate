# src/models.py (c) 2025 Gregory L. Magnusson MIT license

from typing import Optional, Dict, Any
import requests
import logging
import subprocess
from pathlib import Path
import json
from datetime import datetime

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
    
    def check_installation(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            self.error = str(e)
            logger.error(f"Ollama installation check failed: {e}")
            return False
    
    def list_models(self) -> list:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error listing Ollama models: {e}")
            return []
    
    def select_model(self, model_name: str) -> bool:
        """Select Ollama model"""
        try:
            # Check if model exists
            models = self.list_models()
            if model_name not in models:
                # Try to pull the model
                response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                )
                if response.status_code != 200:
                    self.error = f"Failed to pull model {model_name}"
                    return False
            
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
            full_prompt = f"""
            Context: {context}
            
            Question: {prompt}
            
            Answer:""" if context else prompt
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                self.error = f"API Error: {response.text}"
                return f"Error: {response.text}"
            
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
                messages=messages,
                temperature=0.7
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
                model="mixtral-8x7b-32768",
                temperature=0.7
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

class HuggingFaceHandler(BaseHandler):
    """Handler for HuggingFace models"""
    
    def __init__(self):
        super().__init__()
        self.error = "HuggingFace handler not implemented"
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        return "HuggingFace handler not implemented"
