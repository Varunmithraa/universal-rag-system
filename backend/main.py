import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import BASE_DIR, UPLOAD_DIR, SAMPLE_DIR, settings
from backend.parser import DocumentParser
from backend.chunker import SmartChunker
from backend.rag_engine import rag_engine

app = FastAPI(
    title="Universal RAG AI Engine",
    description="High-precision Retrieval-Augmented Generation for any document format",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    doc_filter: Optional[str] = None
    top_k: Optional[int] = 4

class SettingsRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    llm_model: Optional[str] = None

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/stats")
def get_stats():
    return rag_engine.get_stats()

@app.get("/api/documents")
def list_documents():
    return {"documents": rag_engine.list_documents()}

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    success = rag_engine.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or delete failed")
    return {"message": "Document deleted successfully", "doc_id": doc_id}

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    chunker = SmartChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    indexed_results = []

    for file in files:
        filename = file.filename
        doc_id = str(uuid.uuid4())[:8]
        save_path = UPLOAD_DIR / f"{doc_id}_{filename}"

        # Write uploaded file to disk
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = save_path.stat().st_size
        ext = save_path.suffix.lower()

        # Parse document structure
        sections = DocumentParser.parse_file(save_path, filename)
        
        # Split into semantic chunks
        chunks = chunker.chunk_document(doc_id=doc_id, filename=filename, sections=sections)

        # Index in ChromaDB
        rag_engine.add_document_chunks(
            doc_id=doc_id,
            filename=filename,
            file_type=ext,
            file_size_bytes=file_size,
            chunks=chunks
        )

        # Record upload time
        if doc_id in rag_engine.docs_metadata:
            rag_engine.docs_metadata[doc_id]["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            rag_engine._save_metadata()

        indexed_results.append({
            "doc_id": doc_id,
            "filename": filename,
            "file_type": ext,
            "size_bytes": file_size,
            "chunks_count": len(chunks)
        })

    return {
        "message": f"Successfully indexed {len(indexed_results)} document(s)",
        "documents": indexed_results
    }

@app.post("/api/query")
def query_rag(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = rag_engine.query(
        question=req.question.strip(),
        doc_filter=req.doc_filter,
        top_k=req.top_k or 4
    )
    return result

@app.get("/api/settings")
def get_settings():
    masked_key = ""
    if settings.gemini_api_key:
        masked_key = f"{settings.gemini_api_key[:4]}...{settings.gemini_api_key[-4:]}" if len(settings.gemini_api_key) > 8 else "***"
    return {
        "has_gemini_key": bool(settings.gemini_api_key),
        "masked_key": masked_key,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model
    }

@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    if req.gemini_api_key is not None:
        settings.gemini_api_key = req.gemini_api_key.strip()
    if req.llm_model:
        settings.llm_model = req.llm_model.strip()
    return {"message": "Settings updated successfully", "has_key": bool(settings.gemini_api_key), "model": settings.llm_model}

@app.post("/api/load-samples")
def load_sample_documents():
    """Populates the knowledge base with rich multi-format sample documents."""
    chunker = SmartChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    samples = list(SAMPLE_DIR.glob("*.*"))
    indexed = []

    for sample_file in samples:
        filename = sample_file.name
        doc_id = f"sample_{sample_file.stem}"
        dest_path = UPLOAD_DIR / f"{doc_id}_{filename}"
        shutil.copyfile(sample_file, dest_path)

        file_size = dest_path.stat().st_size
        ext = dest_path.suffix.lower()

        sections = DocumentParser.parse_file(dest_path, filename)
        chunks = chunker.chunk_document(doc_id=doc_id, filename=filename, sections=sections)

        rag_engine.add_document_chunks(
            doc_id=doc_id,
            filename=f"[Sample] {filename}",
            file_type=ext,
            file_size_bytes=file_size,
            chunks=chunks
        )
        if doc_id in rag_engine.docs_metadata:
            rag_engine.docs_metadata[doc_id]["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            rag_engine._save_metadata()

        indexed.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": len(chunks)
        })

    return {"message": f"Loaded {len(indexed)} sample documents", "samples": indexed}

# Mount static frontend files
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
