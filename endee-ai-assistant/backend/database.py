"""
database.py
───────────
All Endee vector database operations live here:
  • Initialise / connect to the Endee server
  • Create the knowledge index (idempotent)
  • Upsert vectors with metadata
  • Perform nearest-neighbour similarity search
  • Delete all vectors for a specific document (optional clean-up)

Endee Python SDK: pip install endee
Endee server   : docker run -p 8080:8080 endeeio/endee-server:latest
"""

import logging
import uuid
from typing import List, Dict, Any

from endee import Endee, Precision

from embedding import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
ENDEE_HOST: str = "http://localhost:8080"    # Local Docker instance
INDEX_NAME: str = "knowledge_base"           # Single index for all documents
SPACE_TYPE: str = "cosine"                   # Cosine similarity for text retrieval
TOP_K_DEFAULT: int = 5                       # Default number of results to retrieve

# Endee client — initialised in `init_db()`
_client: Endee | None = None
_index = None


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Connect to the Endee server and ensure the knowledge index exists.
    Call this once at application startup (FastAPI lifespan event).
    """
    global _client, _index

    logger.info(f"Connecting to Endee at {ENDEE_HOST} …")
    _client = Endee()
    _client.set_base_url(f"{ENDEE_HOST}/api/v1")

    # Create the index if it doesn't already exist (idempotent)
    try:
        _client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,   # 384 for all-MiniLM-L6-v2
            space_type=SPACE_TYPE,
            precision=Precision.INT8,        # INT8 quantisation: 4× memory savings
        )
        logger.info(f"Index '{INDEX_NAME}' created.")
    except Exception as e:
        # Endee raises an exception if index already exists — that's fine
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            logger.info(f"Index '{INDEX_NAME}' already exists, reusing it.")
        else:
            raise

    _index = _client.get_index(name=INDEX_NAME)
    logger.info("Endee initialised successfully.")


def _ensure_ready() -> None:
    """Guard: raise if init_db() was not called."""
    if _index is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")

def upsert_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
    _ensure_ready()

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length.")

    records = []
    for chunk, vector in zip(chunks, embeddings):
        record = {
            "id": str(uuid.uuid4()),       # Unique ID per chunk
            "vector": vector,
            "meta": {
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["index"],
            },
        }
        records.append(record)

    # Endee's upsert accepts a list of dicts
    _index.upsert(records)
    logger.info(f"Upserted {len(records)} vectors into '{INDEX_NAME}'.")
    return len(records)


def similarity_search(
    query_vector: List[float],
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    _ensure_ready()

    raw_results = _index.query(vector=query_vector, top_k=top_k)

    results = []

    for item in raw_results:
        meta = item.get("meta", {})

        results.append({
            "id": item.get("id"),
            "score": round(float(item.get("similarity", 0)), 4),
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1),
        })

    return results


def get_index_stats() -> Dict[str, Any]:
    _ensure_ready()
    try:
        info = _index.info()
        return {
            "index_name": INDEX_NAME,
            "dimension": EMBEDDING_DIMENSION,
            "space_type": SPACE_TYPE,
            "vector_count": getattr(info, "vector_count", "n/a"),
        }
    except Exception as e:
        logger.warning(f"Could not retrieve index stats: {e}")
        return {"index_name": INDEX_NAME, "error": str(e)}
