# rage.py (c) 2035 RAGE

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import time
from datetime import datetime
from reasoning import RageReasoning
from models import (
    GPT4Handler, 
    GroqHandler, 
    TogetherHandler, 
    OllamaHandler,
    HuggingFaceHandler
)
from knowledge import KnowledgeBase
from config import model_config
from logger import get_logger

# Initialize logger
logger = get_logger('rage')

class RAGE:
    """RAGE - Retrieval Augmented Generation Environment"""
    
    def __init__(self):
        self.setup_session_state()
        self.knowledge_base = KnowledgeBase()
        self.load_css()
    
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
            .cost-tracker { padding: 10px; background: #f0f2f6; border-radius: 5px; }
            .model-info { padding: 10px; background: #f0f2f6; border-radius: 5px; }
            .capability-tag { 
                display: inline-block; 
                padding: 2px 8px; 
                margin: 2px;
                background: #e1e4e8; 
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
            .checkmark { color: #00c853; font-weight: bold; }
            </style>
        """, unsafe_allow_html=True)
    
    def initialize_model(self, provider: str):
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
                st.session_state.model_instances[provider.lower()] = handler_class()
            
            return st.session_state.model_instances[provider.lower()]
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            st.error(f"Error initializing model: {str(e)}")
            return None
    
    def display_model_info(self):
        """Display current model information"""
        if st.session_state.provider and st.session_state.selected_model:
            model_info = model_config.get_model_info(
                st.session_state.provider.lower(),
                st.session_state.selected_model
            )
            if model_info:
                self.render_model_info(model_info)
    
    def render_model_info(self, model_info):
        """Render model information in sidebar"""
        st.sidebar.markdown("### Model Information")
        st.sidebar.markdown(f"""
        <div class="model-info">
            <p><strong>Model:</strong> {model_info['name']}</p>
            <p><strong>Developer:</strong> {model_info['developer']}</p>
            <p><strong>Max Tokens:</strong> {model_info['tokens']}</p>
            <p><strong>Cost:</strong> {model_info['cost']}</p>
            <div><strong>Capabilities:</strong></div>
            {''.join([f'<span class="capability-tag">{cap}</span>' 
                     for cap in model_info.get('capabilities', [])])}
        </div>
        """, unsafe_allow_html=True)
    
    def process_message(self, prompt):
        """Process and generate response to user message"""
        try:
            if not st.session_state.provider:
                st.warning("Please select an AI Provider first")
                return
                
            model = self.initialize_model(st.session_state.provider)
            if not model:
                return

            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Processing with RAG..."):
                    try:
                        # Get relevant context from knowledge base
                        context = self.knowledge_base.get_relevant_context(prompt)
                        
                        # Generate response using model and context
                        response = model.generate_response(prompt, context)
                        
                        if isinstance(model, OllamaHandler) and model.get_last_error():
                            st.error(model.get_last_error())
                            return
                        
                        st.markdown(response)
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
            
            # Sidebar configuration
            self.setup_sidebar()
            
            # Chat interface
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Chat input
            prompt = st.chat_input(
                "Enter your query...",
                key="chat_input"
            )
            
            if prompt:
                self.process_message(prompt)

        except Exception as e:
            logger.error(f"Main application error: {e}")
            st.error("An error occurred in the application. Please try refreshing the page.")

def main():
    rage = RAGE()
    rage.run()

if __name__ == "__main__":
    main()
