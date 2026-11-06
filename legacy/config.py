"""
Configuration file for API keys and settings.
Loads API keys from .env file or environment variables for security.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#API keys are loaded from .env file (not committed to git)
#Use .env file (recommended 

# Voyage AI API Key 
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY', None)

# OpenRouter API Key
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', None)

# Hugging Face API Key 
HF_API_KEY = os.getenv('HF_API_KEY', None)

# Together AI API Key
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', None)

# API ENDPOINTS 
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_API_URL = "https://api-inference.huggingface.co"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# MODEL NAMES 
VOYAGE_MODEL = "voyage-lite-02-instruct"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" 
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
