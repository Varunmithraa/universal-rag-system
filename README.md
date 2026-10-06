# OmniRAG: Universal Document Intelligence System

A full-stack, enterprise-grade Retrieval-Augmented Generation (RAG) platform that accepts **any type of document** (PDF, Word, PowerPoint, Excel, CSV, JSON, Markdown, Code, Plain Text) and provides **grounded, cited answers** using semantic vector search and Google Gemini.

---

## 🏗️ System Architecture

OmniRAG follows a modular 5-tier architecture:

```
+--------------------------------------------------------------------------+
|                            Web Interface (SPA)                           |
|      Modern Glassmorphism UI (Dashboard, Q&A Chat, Document Hub, Docs)   |
+------------------------------------+-------------------------------------+
                                     |  REST API (FastAPI)
                                     v
+--------------------------------------------------------------------------+
|                       1. Universal Parsing Engine                        |
|  - PDF         -> PyMuPDF (fitz) page extraction                         |
|  - Word        -> python-docx paragraph & table extraction               |
|  - PowerPoint  -> python-pptx slide & note extraction                    |
|  - Excel / CSV -> pandas / openpyxl tabular serialization                |
|  - Text / Code -> Universal multi-encoding auto-decoder                  |
+------------------------------------+-------------------------------------+
                                     |  Structured Section Stream
                                     v
+--------------------------------------------------------------------------+
|                       2. Semantic Chunking Engine                        |
|  - Recursive character splitter (Paragraphs -> Sentences -> Words)       |
|  - 800-character window with 150-character contextual overlap            |
|  - Metadata preservation: doc_id, filename, page_number, section_title   |
+------------------------------------+-------------------------------------+
                                     |  Chunk Batches
                                     v
+--------------------------------------------------------------------------+
|                  3. Vector Store & Embedding Engine                      |
|  - ChromaDB persistent vector database with HNSW approximate search      |
|  - Embedding Engine:                                                     |
|      • Local Engine: all-MiniLM-L6-v2 (Chroma ONNX, 384-dim, 100% offline)|
|      • Cloud Engine: gemini-embedding-2 / text-embedding-004             |
+------------------------------------+-------------------------------------+
                                     |  Top-K Semantic Context Matches
                                     v
+--------------------------------------------------------------------------+
|                  4. Grounded Generation & Citation Engine                |
|  - Model: Google Gemini (gemini-2.5-flash / gemini-2.5-pro)              |
|  - Strict anti-hallucination grounded system instruction                 |
|  - Precise inline citations: [Source X: Document, Page Y]                |
|  - Interactive Inspector: Click citations to inspect raw passage evidence|
+--------------------------------------------------------------------------+
```

---

## 🔄 End-to-End Workflow

1. **Ingestion & Parsing**:
   - The user drags and drops any file into the Document Hub or clicks **Load Samples**.
   - The backend `DocumentParser` identifies the MIME/extension and extracts clean textual structures and page numbers.
2. **Chunking & Metadata Tagging**:
   - The `SmartChunker` divides the text into overlapping segments, preventing contextual fracture at chunk borders.
3. **Vector Indexing**:
   - Vectors are indexed into a persistent ChromaDB collection (`universal_rag_docs`) with metadata tags.
4. **Query & Semantic Retrieval**:
   - When a user asks a question, the query is vectorized and compared against the document database using cosine similarity.
   - The top 4 most pertinent context chunks are gathered with similarity scores.
5. **Synthesis & Citation**:
   - The retrieved excerpts and user question are passed into `gemini-2.5-flash`.
   - The model generates an authoritative answer citing `[Source 1]`, `[Source 2]`, etc.
   - Users can click any citation pill to view the exact text fragment and score.

---

## 🤖 Models & Technologies Used

| Component | Model / Library | Specification |
| :--- | :--- | :--- |
| **Generation LLM** | `gemini-2.5-flash` | Ultra-fast multimodal model, 1M context window, high grounding fidelity. Configurable to `gemini-2.5-pro` or `gemini-3.7-flash`. |
| **Local Vector Embeddings** | `all-MiniLM-L6-v2` | 384-dimensional dense sentence embeddings via Chroma ONNX. Runs locally with zero API key requirement. |
| **Cloud Vector Embeddings** | `gemini-embedding-2` | Google GenAI embedding model for unified multimodal semantics. |
| **Vector Database** | `ChromaDB` | Fast, persistent, embedded vector database utilizing HNSW indexing. |
| **Document Parsers** | `PyMuPDF`, `python-docx`, `python-pptx`, `pandas` | Specialized parsers for PDF, DOCX, PPTX, Excel, CSV, JSON, Markdown, Code. |
| **Backend Framework** | `FastAPI` + `Uvicorn` | Asynchronous high-performance REST API. |
| **Frontend UI** | HTML5, Vanilla CSS, JavaScript | Glassmorphism dark-mode UI with live citations and responsive architecture tabs. |

---

## 🚀 How to Run

1. Open your terminal in this directory:
   ```powershell
   cd "C:\Users\varun\.gemini\antigravity-ide\scratch\universal-rag-system"
   ```
2. Start the application:
   ```powershell
   python start.py
   ```
3. Your browser will automatically open:
   👉 **`http://127.0.0.1:8000`**
