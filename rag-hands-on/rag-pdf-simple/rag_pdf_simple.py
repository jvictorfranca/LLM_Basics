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