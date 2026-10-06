import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.config import settings, CHROMA_DIR

class RAGEngine:
    """
    Central RAG Engine orchestrating ChromaDB vector indexing, semantic search,
    source citations, and answer synthesis via Google Gemini API (gemini-2.5-flash)
    or local extractive synthesis.
    """

    def __init__(self):
        # Initialize persistent ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection_name = "universal_rag_docs"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Universal multi-format RAG document store"}
        )
        # In-memory document registry for fast stats
        self.docs_metadata_file = CHROMA_DIR / "docs_metadata.json"
        self.docs_metadata: Dict[str, Dict[str, Any]] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        if self.docs_metadata_file.exists():
            try:
                with open(self.docs_metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self):
        try:
            with open(self.docs_metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.docs_metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def add_document_chunks(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        chunks: List[Dict[str, Any]]
    ):
        """Indexes all chunks into ChromaDB and registers document metadata."""
        if not chunks:
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "page": int(c.get("page", 1)),
                "section": str(c.get("section", "General")),
                "chunk_index": int(c.get("chunk_index", 0))
            }
            for c in chunks
        ]

        # Add in batches of 100 for efficiency
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size]
            )

        # Register metadata
        self.docs_metadata[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size_bytes,
            "total_chunks": len(chunks),
            "created_at": None  # will be set by caller
        }
        self._save_metadata()

    def delete_document(self, doc_id: str) -> bool:
        """Removes a document and all its chunks from ChromaDB."""
        try:
            # Query IDs with doc_id
            self.collection.delete(where={"doc_id": doc_id})
            if doc_id in self.docs_metadata:
                del self.docs_metadata[doc_id]
                self._save_metadata()
            return True
        except Exception as e:
            print(f"Error deleting doc {doc_id}: {e}")
            return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns list of all indexed documents with metrics."""
        return list(self.docs_metadata.values())

    def get_stats(self) -> Dict[str, Any]:
        """Returns overall system knowledge stats."""
        total_chunks = self.collection.count()
        total_docs = len(self.docs_metadata)
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "embedding_model": "all-MiniLM-L6-v2 (Chroma Neural ONNX)",
            "llm_model": settings.llm_model,
            "has_gemini_key": bool(settings.gemini_api_key)
        }

    def query(
        self,
        question: str,
        doc_filter: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Retrieves relevant context chunks and generates a grounded response.
        """
        if self.collection.count() == 0:
            return {
                "answer": "No documents have been uploaded yet. Please upload a document (PDF, Word, Excel, PPTX, or Text) to ask questions.",
                "sources": [],
                "model_used": "None"
            }

        where_clause = {"doc_id": doc_filter} if doc_filter else None
        
        # Semantic Retrieval
        results = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, self.collection.count()),
            where=where_clause
        )

        retrieved_sources = []
        context_blocks = []

        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else []
            dists = results["distances"][0] if "distances" in results else []
            ids = results["ids"][0] if "ids" in results else []

            for idx, text in enumerate(docs):
                meta = metas[idx] if idx < len(metas) else {}
                dist = dists[idx] if idx < len(dists) else 0.5
                # Approximate cosine similarity score from Euclidean distance: 1 / (1 + dist)
                score = round(max(0.0, 1.0 / (1.0 + float(dist))), 3)

                source_info = {
                    "source_id": idx + 1,
                    "chunk_id": ids[idx] if idx < len(ids) else f"chunk_{idx}",
                    "filename": meta.get("filename", "Unknown Document"),
                    "page": meta.get("page", 1),
                    "section": meta.get("section", "General"),
                    "text": text,
                    "score": score
                }
                retrieved_sources.append(source_info)
                context_blocks.append(
                    f"--- SOURCE [{idx + 1}] (Document: {source_info['filename']}, Page/Section: {source_info['page']}) ---\n{text}"
                )

        combined_context = "\n\n".join(context_blocks)

        # Generate answer using Gemini if API key is present
        api_key = settings.gemini_api_key.strip()
        if api_key:
            answer = self._generate_gemini_answer(question, combined_context, retrieved_sources)
            model_used = settings.llm_model
        else:
            answer = self._generate_local_answer(question, retrieved_sources)
            model_used = "Local Neural Extractor (Add Gemini API Key for Generative Synthesis)"

        return {
            "answer": answer,
            "sources": retrieved_sources,
            "model_used": model_used
        }

    def _generate_gemini_answer(
        self, question: str, context: str, sources: List[Dict[str, Any]]
    ) -> str:
        try:
            from google import genai
            client = genai.Client(api_key=settings.gemini_api_key)

            system_instruction = (
                "You are an expert, highly accurate RAG (Retrieval-Augmented Generation) document assistant.\n"
                "Your objective is to answer the user's question with utmost fidelity strictly based on the provided source excerpts.\n"
                "Guidelines:\n"
                "1. Direct & Factual: Answer concisely and clearly based on the context.\n"
                "2. Inline Citations: Reference the exact source where you found the information using `[Source X]` (e.g. `[Source 1]`, `[Source 2]`).\n"
                "3. Source Attribution: If different sources give conflicting or complementary details, explicitly mention them.\n"
                "4. Unanswered Information: If the context does not contain enough information to answer the question, state honestly that the uploaded document does not mention it, rather than hallucinating.\n"
                "5. Formatting: Use clean markdown with bullet points, bold key terms, and code blocks if applicable."
            )

            prompt = (
                f"DOCUMENT CONTEXT EXCERPTS:\n{context}\n\n"
                f"USER QUESTION: {question}\n\n"
                "Provide a comprehensive, well-structured answer with [Source X] citations:"
            )

            response = client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.2
                }
            )
            if response.text:
                return response.text
            return "Unable to generate an answer from the model. Please check the prompt or document."
        except Exception as e:
            return f"Error contacting Gemini API: {str(e)}\n\nPlease verify your Gemini API key in Settings."

    def _generate_local_answer(
        self, question: str, sources: List[Dict[str, Any]]
    ) -> str:
        if not sources:
            return "No matching content found for your query in the uploaded documents."

        sections_md = []
        for s in sources[:4]:
            snippet = s['text'].strip()
            # Clean preview
            preview = snippet[:400] + ("..." if len(snippet) > 400 else "")
            sections_md.append(
                f"**[Source {s['source_id']}: {s['filename']}] (Page {s['page']} • {int(s['score'] * 100)}% Match)**\n"
                f"> \"{preview}\""
            )

        excerpts_text = "\n\n".join(sections_md)

        msg = (
            f"### 📄 Retrieved Context Excerpts for: *\"{question}\"*\n\n"
            f"{excerpts_text}\n\n"
            "---\n"
            "✨ **To enable full AI generative synthesis, cross-page analysis & answers:**\n"
            "Click the purple **Settings** button in the top right and enter your **Google Gemini API Key** (free at [ai.google.dev](https://aistudio.google.com/apikey)). "
            "Once saved, `gemini-2.5-flash` will automatically analyze these excerpts and write a complete, natural-language synthesized answer!"
        )
        return msg


# Global RAG engine singleton
rag_engine = RAGEngine()
