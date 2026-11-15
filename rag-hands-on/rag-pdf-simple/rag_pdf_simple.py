import streamlit as st
import chromadb
from chromadb. utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2
import uuid

load_dotenv()

CONST_SIZE = 1000
CHUNK_OVERLAP = 200


class SimpleModelSelector:
    """Simple class to handle model selection"""

    def __init__(self):
        # Available LLM models
        self.llm_models = {"openai": "GPT-4", "ollama": "Llama3"}

        # Available embedding models with their dimensions
        self.embedding_models = {
            "openai": {
                "name": "OpenAI Embeddings",
                "dimensions": 1536,
                "model_name": "text-embedding-3-small",
            },
            "chroma": {"name": "Chroma Default", "dimensions": 384, "model_name": None},
            "nomic": {
                "name": "Nomic Embed Text",
                "dimensions": 768,
                "model_name": "nomic-embed-text",
            },
        }

    def select_models(self):
        """Let user select models through Streamlit UI"""
        st.sidebar.title("📚 Model Selection")

        # Select LLM
        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )

        # Select Embeddings
        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )

        return llm, embedding
