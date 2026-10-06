import re
from typing import List, Dict, Any

class SmartChunker:
    """
    Splits document sections into semantic chunks with overlap while preserving
    metadata (document ID, filename, page, section header, chunk index).
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, doc_id: str, filename: str, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        all_chunks = []
        chunk_counter = 0

        for sec in sections:
            text = sec.get("text", "").strip()
            page = sec.get("page", 1)
            section_title = sec.get("section", "General")

            if not text:
                continue

            sub_chunks = self._recursive_split(text, self.chunk_size, self.chunk_overlap)
            for sub_text in sub_chunks:
                clean_chunk = sub_text.strip()
                if len(clean_chunk) < 20:
                    continue  # Skip negligible fragments
                
                chunk_id = f"{doc_id}_c{chunk_counter}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page,
                    "section": section_title,
                    "chunk_index": chunk_counter,
                    "text": clean_chunk,
                    "char_count": len(clean_chunk)
                })
                chunk_counter += 1

        return all_chunks

    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        return self._split_text(text, separators, chunk_size, chunk_overlap)

    def _split_text(
        self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        
        for sep in separators:
            if sep in text:
                separator = sep
                break

        splits = text.split(separator)
        current_chunk = []
        current_length = 0

        for piece in splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_length + piece_len > chunk_size and current_chunk:
                merged = separator.join(current_chunk)
                final_chunks.append(merged)
                
                # Keep overlap from the end
                overlap_text = merged[-chunk_overlap:] if chunk_overlap > 0 else ""
                current_chunk = [overlap_text, piece] if overlap_text else [piece]
                current_length = len(separator.join(current_chunk))
            else:
                current_chunk.append(piece)
                current_length += piece_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks
