import os
import json
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

SETTINGS_FILE = DATA_DIR / "settings.json"

class Settings(BaseModel):
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    llm_model: str = "gemini-3.7-flash"
    embedding_model: str = "gemini-embedding-2" # Or local ONNX all-MiniLM-L6-v2 fallback
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4

    def save(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.model_dump(), f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    @classmethod
    def load(cls):
        inst = cls()
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("gemini_api_key"):
                        inst.gemini_api_key = data["gemini_api_key"]
                    if data.get("llm_model"):
                        model = data["llm_model"]
                        if "2.5-flash" in model or "2.5-pro" in model:
                            model = "gemini-3.7-flash"
                        inst.llm_model = model
            except Exception as e:
                print(f"Error loading settings: {e}")
        if "2.5-flash" in inst.llm_model or "2.5-pro" in inst.llm_model:
            inst.llm_model = "gemini-3.7-flash"
        return inst

settings = Settings.load()
