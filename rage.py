# rage.py (c) 2025 Gregory L. Magnusson MIT license

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Optional
from src.models import (
    GPT4Handler, 
    GroqHandler, 
    TogetherHandler, 
    OllamaHandler,
    HuggingFaceHandler
)
from src.memory import (
    memory_manager,
    DialogEntry,
    MemoryEntry,
    store_conversation,
    get_conversation_history
)
from src.config import get_config, get_model_config
from src.logger import get_logger
from src.openmind import OpenMind

# Initialize logger
logger = get_logger('rage')

class RAGE:
    """RAGE - Retrieval Augmented Generative Engine"""
    
    def __init__(self):
        self.setup_session_state()
        self.config = get_config()
        self.model_config = get_model_config()
        self.load_css()
        
        # Initialize systems
        self.memory = memory_manager
        self.openmind = OpenMind()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if 'provider' not in st.session_state:
            st.session_state.provider = None
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        if 'model_capabilities' not in st.session_state:
            st.session_state.model_capabilities = []
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
    
    def check_ollama_status(self):
        """Check Ollama installation and available models"""
        try:
            if not st.session_state.model_instances['ollama']:
                st.session_state.model_instances['ollama'] = OllamaHandler()
            
            if st.session_state.model_instances['ollama'].check_installation():
                models = st.session_state.model_instances['ollama'].list_models()
                return True, models
            return False, []
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return False, []
    
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
            .capability-tag { 
                display: inline-block; 
                padding: 2px 8px; 
                margin: 2px;
                background: #3B3B3B; 
                border-radius: 12px; 
                font-size: 0.8em; 
            }
            .api-key-status {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 5px;
                margin: 5px 0;
            }
            .checkmark {
                color: #00cc00;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def initialize_model(self, provider: str) -> Optional[Any]:
        """Initialize or retrieve model instance"""
        try:
            if not provider:
                st.info("Please select an AI Provider")
                return None
            
            if provider == "Together":
                key = self.openmind.get_api_key('together')
                if key:
                    if not st.session_state.model_instances['together']:
                        st.session_state.model_instances['together'] = TogetherHandler(key)
                    return st.session_state.model_instances['together']
                else:
                    st.error("Together API key not found")
                    return None
            
            elif provider == "Groq":
                key = self.openmind.get_api_key('groq')
                if key:
                    if not st.session_state.model_instances['groq']:
                        st.session_state.model_instances['groq'] = GroqHandler(key)
                    return st.session_state.model_instances['groq']
                else:
                    st.error("Groq API key not found")
                    return None
            
            elif provider == "OpenAI":
                key = self.openmind.get_api_key('openai')
                if key:
                    if not st.session_state.model_instances['openai']:
                        st.session_state.model_instances['openai'] = GPT4Handler(key)
                    return st.session_state.model_instances['openai']
                else:
                    st.error("OpenAI API key not found")
                    return None
            
            elif provider == "Ollama":
                if not st.session_state.model_instances['ollama']:
                    st.session_state.model_instances['ollama'] = OllamaHandler()
                
                if st.session_state.model_instances['ollama'].check_installation():
                    available_models = st.session_state.model_instances['ollama'].list_models()
                    if available_models:
                        if not st.session_state.selected_model:
                            st.info("Please select an Ollama model to continue")
                            return None
                        
                        if st.session_state.model_instances['ollama'].select_model(st.session_state.selected_model):
                            return st.session_state.model_instances['ollama']
                        else:
                            st.error(st.session_state.model_instances['ollama'].get_last_error())
                            return None
                    else:
                        st.error("No Ollama models found. Please pull a model first.")
                        return None
                else:
                    st.error("Ollama service is not running. Please start the Ollama service.")
                    return None
            
            elif provider == "HuggingFace":
                if not st.session_state.model_instances['huggingface']:
                    st.session_state.model_instances['huggingface'] = HuggingFaceHandler()
                return st.session_state.model_instances['huggingface']
            
            return None
            
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
                with st.spinner("Processing with RAGE..."):
                    try:
                        # Get relevant context
                        context = self.memory.get_relevant_context(prompt)
                        
                        # Generate response
                        response = model.generate_response(prompt, context)
                        
                        if isinstance(model, OllamaHandler) and model.get_last_error():
                            st.error(model.get_last_error())
                            return
                        
                        # Store conversation
                        dialog_entry = DialogEntry(
                            query=prompt,
                            response=response,
                            provider=st.session_state.provider,
                            model=st.session_state.selected_model,
                            context={"retrieved_context": context}
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
            
            # Check Ollama status
            ollama_running, ollama_models = self.check_ollama_status()
            if ollama_running:
                st.markdown("""
                    <div class="api-key-status">
                        <span class="checkmark">●</span>
                        <span class="text">Ollama Running</span>
                    </div>
                    """, unsafe_allow_html=True)
                if ollama_models:
                    st.caption(f"Available models: {', '.join(ollama_models)}")
            
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
                if st.session_state.provider == "Ollama":
                    if ollama_models:
                        st.session_state.selected_model = st.selectbox(
                            "Select Ollama Model",
                            options=ollama_models,
                            key='ollama_model_select'
                        )
                else:
                    provider_models = self.model_config.get_provider_models(
                        st.session_state.provider.lower()
                    )
                    if provider_models:
                        st.session_state.selected_model = st.selectbox(
                            f"Select {st.session_state.provider} Model",
                            options=list(provider_models.keys()),
                            key=f"{st.session_state.provider.lower()}_model_select"
                        )
                        
                        # API key handling
                        if st.session_state.provider in ["OpenAI", "Together", "Groq"]:
                            # Check if API key exists
                            existing_key = self.openmind.get_api_key(
                                st.session_state.provider.lower()
                            )
                            
                            # Show API key status
                            if existing_key:
                                st.markdown(f"""
                                    <div class="api-key-status">
                                        <span class="checkmark">✓</span>
                                        <span class="text">{st.session_state.provider} API Key Stored</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # API key input
                            api_key = st.text_input(
                                f"{st.session_state.provider} API Key",
                                type="password",
                                key=f"{st.session_state.provider.lower()}_api_key"
                            )
                            
                            if api_key:
                                self.openmind.save_api_key(
                                    st.session_state.provider.lower(), 
                                    api_key
                                )
                                try:
                                    st.rerun()
                                except AttributeError:
                                    try:
                                        st.experimental_rerun()
                                    except AttributeError:
                                        st.empty()
            
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
            st.title("RAGE - Retrieval Augmented Generative Engine")
            
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
