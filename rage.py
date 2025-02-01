# rage.py (c) 2025 Gregory L. Magnusson MIT license

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Optional
from models import (
    GPT4Handler, 
    GroqHandler, 
    TogetherHandler, 
    OllamaHandler,
    HuggingFaceHandler
)
from memory import (
    memory_manager,
    DialogEntry,
    MemoryEntry,
    store_conversation,
    get_conversation_history
)
from config import get_config, get_model_config
from logger import get_logger

# Initialize logger
logger = get_logger('rage')

class RAGE:
    """RAGE - Retrieval Augmented Generation Environment"""
    
    def __init__(self):
        self.setup_session_state()
        self.config = get_config()
        self.model_config = get_model_config()
        self.load_css()
        
        # Initialize memory system
        self.memory = memory_manager
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if 'provider' not in st.session_state:
            st.session_state.provider = None
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        if 'cost_tracking' not in st.session_state:
            st.session_state.cost_tracking = {"total": 0.0, "session": 0.0}
        if 'model_instances' not in st.session_state:
            st.session_state.model_instances = {
                'ollama': None,
                'groq': None,
                'together': None,
                'openai': None,
                'huggingface': None
            }
    
    def load_css(self):
        """Load CSS styling"""
        try:
            with open('styles.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error loading CSS: {e}")
            self.load_default_css()
    
    def load_default_css(self):
        """Load default CSS if custom CSS fails"""
        st.markdown("""
            <style>
            .cost-tracker { padding: 10px; background: #262730; border-radius: 5px; }
            .model-info { padding: 10px; background: #1E1E1E; border-radius: 5px; }
            </style>
        """, unsafe_allow_html=True)
    
    def initialize_model(self, provider: str) -> Optional[Any]:
        """Initialize or retrieve model instance"""
        try:
            if not provider:
                st.info("Please select an AI Provider")
                return None
            
            handlers = {
                "OpenAI": GPT4Handler,
                "Together": TogetherHandler,
                "Groq": GroqHandler,
                "Ollama": OllamaHandler,
                "HuggingFace": HuggingFaceHandler
            }
            
            if provider not in handlers:
                st.error(f"Unsupported provider: {provider}")
                return None
            
            if not st.session_state.model_instances[provider.lower()]:
                handler_class = handlers[provider]
                if provider in ["OpenAI", "Together", "Groq"]:
                    api_key = st.session_state.get(f"{provider.lower()}_api_key")
                    if not api_key:
                        st.error(f"{provider} API key required")
                        return None
                    st.session_state.model_instances[provider.lower()] = handler_class(api_key)
                else:
                    st.session_state.model_instances[provider.lower()] = handler_class()
            
            return st.session_state.model_instances[provider.lower()]
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            st.error(f"Error initializing model: {str(e)}")
            return None
    
    def update_cost_tracking(self, response_length: int):
        """Update cost tracking based on usage"""
        try:
            if st.session_state.provider and st.session_state.selected_model:
                model_info = self.model_config.get_model_info(
                    st.session_state.provider.lower(),
                    st.session_state.selected_model
                )
                if model_info and 'cost' in model_info:
                    cost_str = model_info['cost']
                    if '/1M tokens' in cost_str:
                        base_cost = float(cost_str.split('$')[1].split('/')[0])
                        tokens = response_length / 4
                        cost = (tokens / 1000000) * base_cost
                    elif '/1K tokens' in cost_str:
                        base_cost = float(cost_str.split('$')[1].split('/')[0])
                        tokens = response_length / 4
                        cost = (tokens / 1000) * base_cost
                    else:
                        cost = 0.0
                    
                    st.session_state.cost_tracking["session"] += cost
                    st.session_state.cost_tracking["total"] += cost
        except Exception as e:
            logger.error(f"Error updating cost tracking: {e}")
    
    def process_message(self, prompt: str):
        """Process user message and generate response"""
        try:
            if not st.session_state.provider:
                st.warning("Please select an AI Provider first")
                return
            
            model = self.initialize_model(st.session_state.provider)
            if not model:
                return
            
            # Add message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Processing with RAG..."):
                    try:
                        # Generate response
                        response = model.generate_response(prompt)
                        
                        if isinstance(model, OllamaHandler) and model.get_last_error():
                            st.error(model.get_last_error())
                            return
                        
                        # Store conversation in memory
                        dialog_entry = DialogEntry(
                            query=prompt,
                            response=response,
                            provider=st.session_state.provider,
                            model=st.session_state.selected_model
                        )
                        store_conversation(dialog_entry)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Update tracking
                        self.update_cost_tracking(len(response))
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        st.error(f"Error generating response: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            st.error("An error occurred while processing your message")
    
    def setup_sidebar(self):
        """Setup sidebar configuration"""
        with st.sidebar:
            st.header("RAGE Configuration")
            
            # Provider selection
            previous_provider = st.session_state.provider
            st.session_state.provider = st.selectbox(
                "Select AI Provider",
                [None, "OpenAI", "Together", "Groq", "Ollama", "HuggingFace"],
                format_func=lambda x: "Select Provider" if x is None else x
            )
            
            if previous_provider != st.session_state.provider:
                st.session_state.selected_model = None
            
            # Model selection
            if st.session_state.provider:
                provider_models = self.model_config.get_provider_models(
                    st.session_state.provider.lower()
                )
                if provider_models:
                    st.session_state.selected_model = st.selectbox(
                        f"Select {st.session_state.provider} Model",
                        options=list(provider_models.keys()),
                        key=f"{st.session_state.provider.lower()}_model_select"
                    )
                    
                    # API key input for relevant providers
                    if st.session_state.provider in ["OpenAI", "Together", "Groq"]:
                        api_key = st.text_input(
                            f"{st.session_state.provider} API Key",
                            type="password",
                            key=f"{st.session_state.provider.lower()}_api_key"
                        )
            
            # Display model information
            if st.session_state.provider and st.session_state.selected_model:
                model_info = self.model_config.get_model_info(
                    st.session_state.provider.lower(),
                    st.session_state.selected_model
                )
                if model_info:
                    st.markdown("### Model Information")
                    st.markdown(f"""
                    <div class="model-info">
                        <p><strong>Model:</strong> {model_info.name}</p>
                        <p><strong>Developer:</strong> {model_info.developer}</p>
                        <p><strong>Max Tokens:</strong> {model_info.tokens}</p>
                        <p><strong>Cost:</strong> {model_info.cost}</p>
                        <div><strong>Capabilities:</strong></div>
                        {''.join([f'<span class="capability-tag">{cap}</span>' 
                                for cap in model_info.capabilities])}
                    </div>
                    """, unsafe_allow_html=True)
    
    def run(self):
        """Run the RAGE interface"""
        try:
            st.title("RAGE - Retrieval Augmented Generation Environment")
            
            # Display cost tracker
            st.markdown(f"""
                <div class="cost-tracker">
                    Session Cost: ${st.session_state.cost_tracking['session']:.4f}<br>
                    Total Cost: ${st.session_state.cost_tracking['total']:.4f}
                </div>
            """, unsafe_allow_html=True)
            
            # Setup sidebar
            self.setup_sidebar()
            
            # Chat interface
            chat_container = st.container()
            
            with chat_container:
                # Display conversation history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Enter your query..."):
                self.process_message(prompt)
            
        except Exception as e:
            logger.error(f"Main application error: {e}")
            st.error("An error occurred in the application. Please try refreshing the page.")

def main():
    rage = RAGE()
    rage.run()

if __name__ == "__main__":
    main()
