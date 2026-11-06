"""
Converts user text into numerical embeddings using Voyage AI API.
"""

import numpy as np
import os
import requests

try:
    from config import VOYAGE_API_KEY, VOYAGE_API_URL, VOYAGE_MODEL
except ImportError:
    VOYAGE_API_KEY = None
    VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
    VOYAGE_MODEL = "voyage-lite-02-instruct"


class NLPAgent:
    
    def __init__(self, api_key=None):
        """
        Initialize the Voyage AI embedding model.
        
        Args:
            api_key: Voyage AI API key (or set in config.py, or VOYAGE_API_KEY environment variable)
        """
        print("Initializing NLP Agent (Voyage AI API)...")
        
        # Priority: passed parameter > config.py > environment variable
        self.api_key = api_key or VOYAGE_API_KEY or os.getenv('VOYAGE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Voyage API key not found. Please:\n"
                "1. Set VOYAGE_API_KEY in config.py, or\n"
                "2. Set VOYAGE_API_KEY environment variable, or\n"
                "3. Pass api_key parameter"
            )
        
        self.model_name = VOYAGE_MODEL
        self.api_url = VOYAGE_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print("NLP Agent initialized successfully!")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector using Voyage AI API.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array containing the embedding
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Call Voyage AI API
        payload = {
            "model": self.model_name,
            "input": [text]
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Voyage API error: {response.status_code} - {response.text}")
        
        data = response.json()
        embedding = np.array(data['data'][0]['embedding'])
        
        return embedding
    
    def generate_embeddings_batch(self, texts: list) -> np.ndarray:
        """
        Convert multiple texts to embeddings (batch processing).
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (batch_size, embedding_dim) containing embeddings
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Call Voyage AI API
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Voyage API error: {response.status_code} - {response.text}")
        
        data = response.json()
        embeddings = np.array([item['embedding'] for item in data['data']])
        
        return embeddings

