import os
import json
import socket
from pathlib import Path
from pydantic import BaseModel

# Optimize socket resolution on Windows to avoid IPv6 connection timeouts (3+ minute hang)
_orig_getaddrinfo = socket.getaddrinfo
def _getaddrinfo_ipv4_preferred(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = _getaddrinfo_ipv4_preferred

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
    llm_model: str = "gemini-3.5-flash-lite"
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
                        # Migrate deprecated or overloaded models to fast flash-lite
                        if any(x in model for x in ["2.5-flash", "2.5-pro", "3.7-flash"]):
                            model = "gemini-3.5-flash-lite"
                        inst.llm_model = model
            except Exception as e:
                print(f"Error loading settings: {e}")
        if any(x in inst.llm_model for x in ["2.5-flash", "2.5-pro", "3.7-flash"]):
            inst.llm_model = "gemini-3.5-flash-lite"
        return inst

settings = Settings.load()
