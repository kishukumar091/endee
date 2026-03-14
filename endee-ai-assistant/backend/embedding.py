"""
embedding.py
────────────
Handles generation of dense vector embeddings using sentence-transformers.

By default uses 'all-MiniLM-L6-v2' (384-dimensional, fast, high quality)
which runs entirely locally — no API key required.

The module is designed as a singleton so the model is loaded only once.
"""

import logging
from typing import List, Union

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "all-MiniLM-L6-v2"  # 384-dim, ~22 MB download, Apache 2.0
EMBEDDING_DIMENSION: int = 384         # Must match Endee index dimension


# ──────────────────────────────────────────────────────────────────────────────
# Singleton model instance — loaded lazily on first use
# ──────────────────────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Return the cached SentenceTransformer model, loading it if necessary.
    Thread-safe for read-heavy workloads (FastAPI startup loads it eagerly).
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME} …")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _model


def generate_embedding(text: str) -> List[float]:
    """
    Generate a single dense embedding vector for the given text.

    Args:
        text: Input string to embed.

    Returns:
        List of floats with length == EMBEDDING_DIMENSION (384).
    """
    model = get_model()
    # normalize_embeddings=True ensures cosine similarity == dot product
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Generate embeddings for a list of texts efficiently in batches.

    Args:
        texts     : List of input strings.
        batch_size: How many texts to encode per forward pass.

    Returns:
        List of embedding vectors (same order as input texts).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,  # progress bar for large uploads
    )
    return [emb.tolist() for emb in embeddings]
