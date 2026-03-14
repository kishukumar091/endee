import logging
import os
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv

from embedding import generate_embedding
from database import similarity_search

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "openai").lower()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
HF_MODEL_NAME: str = os.getenv("HF_MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
TOP_K: int = 5
MAX_CONTEXT_CHARS: int = 3000   # Safety cap to stay within LLM context window


# ──────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────────────────────────────────────

def _call_openai(system_prompt: str, user_prompt: str) -> str:
    """
    Send a chat-completion request to OpenAI and return the response text.

    Requires OPENAI_API_KEY to be set in the environment / .env file.
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,      # Lower temperature = more factual, less creative
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _call_huggingface(system_prompt: str, user_prompt: str) -> str:
    """
    Inference via HuggingFace Inference API (free tier).
    Falls back gracefully if the model is loading ("estimated time").
    """
    import httpx

    hf_token = os.getenv("HF_API_TOKEN", "")
    combined_prompt = f"{system_prompt}\n\nUser question: {user_prompt}\n\nAnswer:"

    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    payload = {
        "inputs": combined_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "return_full_text": False,
        },
    }

    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"
    with httpx.Client(timeout=60) as client:
        resp = client.post(api_url, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()

    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").strip()
    elif isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"HuggingFace API error: {result['error']}")
    return str(result)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Route to the configured LLM backend."""
    if LLM_BACKEND == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file "
                "or switch LLM_BACKEND=huggingface."
            )
        return _call_openai(system_prompt, user_prompt)
    elif LLM_BACKEND == "huggingface":
        return _call_huggingface(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_BACKEND='{LLM_BACKEND}'. Choose 'openai' or 'huggingface'.")


# ──────────────────────────────────────────────────────────────────────────────
# Context assembly
# ──────────────────────────────────────────────────────────────────────────────

def _build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a readable context block for the LLM prompt.

    Each chunk is prefixed with its source filename and similarity score,
    helping the model understand which document the information came from.
    """
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(retrieved_chunks, start=1):
        header = f"[Source {i}: {chunk['source']} | relevance: {chunk['score']:.2%}]"
        block = f"{header}\n{chunk['text']}"

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            # Truncate if we're about to overflow
            remaining = MAX_CONTEXT_CHARS - total_chars
            block = block[:remaining] + " …"
            context_parts.append(block)
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n".join(context_parts)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Step 1 & 2 of the RAG pipeline:
      - Embed the user's question.
      - Query Endee for the top-k most similar document chunks.

    Args:
        question: Raw user question string.
        top_k   : Number of chunks to retrieve.

    Returns:
        List of chunk dicts with fields: id, score, text, source, chunk_index.
    """
    logger.info(f"Generating query embedding for: '{question[:80]}…'")
    query_embedding = generate_embedding(question)

    logger.info(f"Searching Endee for top-{top_k} chunks …")
    chunks = similarity_search(query_vector=query_embedding, top_k=top_k)

    logger.info(f"Retrieved {len(chunks)} chunks (top score: {chunks[0]['score'] if chunks else 'N/A'})")
    return chunks


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Steps 3-5 of the RAG pipeline:
      - Build context from retrieved chunks.
      - Send context + question to the LLM.
      - Return the answer and the source citations.

    Args:
        question        : User's original question.
        retrieved_chunks: Chunks returned by retrieve_relevant_chunks().

    Returns:
        Tuple of (answer_text, sources_list).
    """
    if not retrieved_chunks:
        return (
            "I couldn't find any relevant information in the uploaded documents. "
            "Please upload a document that contains information related to your question.",
            [],
        )

    context = _build_context(retrieved_chunks)

    system_prompt = (
        "You are an expert AI assistant that answers questions strictly based on "
        "the provided document context. Follow these rules:\n"
        "1. Answer ONLY using the information in the context below.\n"
        "2. If the context doesn't contain the answer, say so clearly.\n"
        "3. Be concise, accurate, and cite the source document when relevant.\n"
        "4. Use markdown formatting for clarity (lists, bold terms, etc.).\n\n"
        f"=== DOCUMENT CONTEXT ===\n{context}\n=== END OF CONTEXT ==="
    )

    user_prompt = f"Question: {question}"

    logger.info(f"Calling LLM backend: {LLM_BACKEND.upper()} …")
    answer = _call_llm(system_prompt, user_prompt)

    # Build clean source citations for the frontend
    sources = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "score": c["score"],
            "preview": c["text"][:200] + ("…" if len(c["text"]) > 200 else ""),
        }
        for c in retrieved_chunks
    ]

    return answer, sources


def run_rag_pipeline(question: str) -> Dict[str, Any]:
    """
    Full end-to-end RAG pipeline entry point.

    Args:
        question: User's question string.

    Returns:
        Dict with keys: answer, sources, top_k_retrieved.
    """
    chunks = retrieve_relevant_chunks(question, top_k=TOP_K)
    answer, sources = generate_answer(question, chunks)

    return {
        "answer": answer,
        "sources": sources,
        "top_k_retrieved": len(chunks),
    }
