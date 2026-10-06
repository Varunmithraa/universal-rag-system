import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
SAMPLE_DIR = BASE_DIR / "sample_docs"

# Create required directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-2" # Or local ONNX all-MiniLM-L6-v2 fallback
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4

settings = Settings()
