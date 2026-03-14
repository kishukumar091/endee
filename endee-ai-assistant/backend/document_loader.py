"""
document_loader.py
──────────────────
Responsible for:
  1. Parsing uploaded files (PDF or TXT).
  2. Extracting raw text content.
  3. Splitting text into overlapping chunks suitable for embedding.

Dependencies: PyMuPDF (fitz), langchain_text_splitters
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ──────────────────────────────────────────────────────────────────────────────
# Configuration — tune these for your use-case
# ──────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512          # Max characters per chunk
CHUNK_OVERLAP: int = 64        # Overlap between consecutive chunks to preserve context


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    full_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"[Page {page_num}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(full_text)


def _extract_text_from_txt(file_path: str) -> str:
    """Read plain text from a .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """
    Dispatch to the correct extractor based on file extension.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        Extracted raw text as a single string.

    Raises:
        ValueError: If the file format is not supported.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(file_path)
    elif suffix == ".txt":
        return _extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: '{suffix}'. Supported: .pdf, .txt")


def chunk_text(text: str, source_name: str) -> List[Dict[str, Any]]:
    """
    Split raw text into overlapping chunks and attach metadata.

    Each chunk is a dict with:
      - text  : the chunk content
      - source: original file name
      - index : chunk sequence number (0-based)

    Args:
        text       : Full document text.
        source_name: Human-readable document identifier (e.g., filename).

    Returns:
        List of chunk dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Smart separators: paragraph → sentence → word → character
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks: List[str] = splitter.split_text(text)

    chunks = [
        {
            "text": chunk.strip(),
            "source": source_name,
            "index": i,
        }
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()  # discard whitespace-only chunks
    ]

    return chunks


def load_and_chunk_document(file_path: str) -> List[Dict[str, Any]]:
    """
    End-to-end pipeline: extract text ➜ split into chunks.

    Args:
        file_path: Path to the uploaded document.

    Returns:
        List of chunk dicts (text, source, index).
    """
    source_name = Path(file_path).name
    raw_text = extract_text(file_path)
    chunks = chunk_text(raw_text, source_name)
    return chunks
