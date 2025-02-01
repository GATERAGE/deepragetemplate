# /src/generator.py (c) 2025 RAGE

from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import yaml
import requests
from .retriever import EnhancedRetriever

class OllamaClient:
    """Ollama API client"""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.error = None
    
    def generate(self, 
                prompt: str,
                temperature: float = 0.7,
                max_tokens: int = 500) -> str:
        """Generate response using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            self.error = str(e)
            return f"Error: {str(e)}"
    
    def get_last_error(self) -> Optional[str]:
        return self.error

class DeepSeekRAGE:
    """Main RAG system"""
    
    def __init__(self, 
                 model_name: str = "deepseek-coder:6.7b",
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 knowledge_dir: str = "./knowledge"):
        self.llm = OllamaClient(model=model_name)
        self.retriever = EnhancedRetriever(knowledge_dir=knowledge_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def load_knowledge_base(self, directory: Optional[str] = None):
        """Load knowledge base"""
        directory = directory or self.retriever.knowledge_dir
        self.retriever.load_directory(directory)
        self.retriever.create_index(
            list(self.retriever.document_store.documents.values())
        )
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response with context"""
        retrieved_docs = self.retriever.retrieve(query)
        context = "\n".join([doc.content for doc in retrieved_docs])
        
        prompt = f"""
        Based on the following context, provide a detailed and accurate response.
        If information is not available in the context, clearly state that.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            "response": response.strip(),
            "sources": [doc.metadata for doc in retrieved_docs],
            "timestamp": datetime.now().isoformat()
        }
    
    def save_response(self, 
                     query: str, 
                     response: Dict[str, Any], 
                     format: str = 'json'):
        """Save response in specified format"""
        timestamp = datetime.now().isoformat()
        output_dir = self.retriever.knowledge_dir / "responses"
        output_dir.mkdir(exist_ok=True)
        
        data = {
            "query": query,
            "response": response,
            "timestamp": timestamp
        }
        
        if format == 'json':
            self._save_response_json(data, output_dir, timestamp)
        elif format == 'markdown':
            self._save_response_markdown(data, output_dir, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_response_json(self, data: Dict, output_dir: Path, timestamp: str):
        """Save response as JSON"""
        output_path = output_dir / f"response_{timestamp}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _save_response_markdown(self, data: Dict, output_dir: Path, timestamp: str):
        """Save response as markdown"""
        output_path = output_dir / f"response_{timestamp}.md"
        content = f"""---
timestamp: {timestamp}
query: {data['query']}
---

# Response

{data['response']['response']}

## Sources Used

{yaml.dump(data['response']['sources'])}
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
