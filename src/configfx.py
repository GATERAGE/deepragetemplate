# src/configfx.py
from pathlib import Path
import yaml
import streamlit as st
from PIL import Image
import base64

class UIConfig:
    """UI Configuration Manager for RAGE"""
    
    def __init__(self, config_file: str = "src/configfx.yaml"):
        self.config_path = Path(config_file)
        self.base_path = self.config_path.parent.parent
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Error loading UI config: {e}")
            return {}
    
    def get_system_prompt(self) -> str:
        """Load system prompt from file"""
        try:
            prompt_path = self.base_path / self.config['prompts']['system_prompt_file']
            with open(prompt_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            st.error(f"Error loading system prompt: {e}")
            return ""
    
    def set_page_config(self):
        """Configure Streamlit page"""
        try:
            favicon_path = self.base_path / self.config['branding']['favicon']
            st.set_page_config(
                page_title=self.config['branding']['title'],
                page_icon=str(favicon_path),
                layout="wide"
            )
        except Exception as e:
            st.error(f"Error setting page config: {e}")
    
    def get_logo_html(self) -> str:
        """Get HTML for logo display"""
        try:
            logo_path = self.base_path / self.config['branding']['logo']
            with open(logo_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f'<img src="data:image/png;base64,{data}" style="max-width: 200px;">'
        except Exception as e:
            st.error(f"Error loading logo: {e}")
            return ""
    
    def get_chat_icons(self) -> tuple:
        """Get user and assistant icons"""
        try:
            user_icon = Image.open(
                self.base_path / self.config['branding']['user_icon']
            )
            assistant_icon = Image.open(
                self.base_path / self.config['branding']['assistant_icon']
            )
            return user_icon, assistant_icon
        except Exception as e:
            st.error(f"Error loading chat icons: {e}")
            return None, None

# Global instance
ui_config = UIConfig()
