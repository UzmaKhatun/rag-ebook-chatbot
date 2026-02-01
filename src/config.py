# """
# Configuration management for the RAG Chatbot
# """
# import os
# from pathlib import Path
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# class Config:
#     """Central configuration class"""
    
#     # Project paths
#     BASE_DIR = Path(__file__).parent.parent
#     DATA_DIR = BASE_DIR / "data"
#     PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"
    
#     # Groq API
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
#     # ChromaDB
#     CHROMA_PERSIST_DIRECTORY = os.getenv(
#         "CHROMA_PERSIST_DIRECTORY", 
#         str(DATA_DIR / "chroma_db")
#     )
#     CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "agentic_ai_ebook")
    
#     # Embedding Model
#     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
#     # LLM Configuration
#     LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
#     LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
#     LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
#     # Chunking Configuration
#     CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
#     CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
#     # Retrieval Configuration
#     TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
#     SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
#     @classmethod
#     def validate(cls):
#         """Validate essential configuration"""
#         if not cls.GROQ_API_KEY:
#             raise ValueError("GROQ_API_KEY not found in environment variables")
        
#         if not cls.PDF_PATH.exists():
#             raise FileNotFoundError(f"PDF not found at {cls.PDF_PATH}")
        
#         # Create directories if they don't exist
#         cls.DATA_DIR.mkdir(exist_ok=True, parents=True)
#         Path(cls.CHROMA_PERSIST_DIRECTORY).mkdir(exist_ok=True, parents=True)
        
#         return True

# # Validate on import
# try:
#     Config.validate()
#     print("✅ Configuration validated successfully")
# except Exception as e:
#     print(f"⚠️ Configuration validation warning: {e}")





"""
Configuration management for the RAG Chatbot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"
    
    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # ChromaDB
    CHROMA_PERSIST_DIRECTORY = os.getenv(
        "CHROMA_PERSIST_DIRECTORY", 
        str(DATA_DIR / "chroma_db")
    )
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "agentic_ai_ebook")
    
    # Embedding Model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # LLM Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
    # Chunking Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    
    @classmethod
    def validate(cls):
        """Validate essential configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        if not cls.PDF_PATH.exists():
            raise FileNotFoundError(f"PDF not found at {cls.PDF_PATH}")
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True, parents=True)
        Path(cls.CHROMA_PERSIST_DIRECTORY).mkdir(exist_ok=True, parents=True)
        
        return True

# Validate on import
try:
    Config.validate()
    print("✅ Configuration validated successfully")
except Exception as e:
    print(f"⚠️ Configuration validation warning: {e}")