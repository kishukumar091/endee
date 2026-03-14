import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import get_index_stats, init_db, upsert_chunks
from document_loader import load_and_chunk_document
from embedding import generate_embeddings_batch, get_model
from rag_pipeline import run_rag_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Load environment variables from .env (if present)
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Allowed file types
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 20


# ──────────────────────────────────────────────────────────────────────────────
# Startup / Shutdown lifecycle
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Runs startup logic before the server accepts requests, and shutdown logic
    after it stops.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("🚀 Starting AI Knowledge Assistant …")

    # Pre-load the embedding model into memory (avoids cold-start on first request)
    logger.info("Loading sentence-transformer model …")
    get_model()

    # Connect to Endee and initialise the knowledge index
    init_db()

    logger.info("✅ Server ready.")
    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("👋 Shutting down AI Knowledge Assistant.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Knowledge Assistant",
    description=(
        "RAG-powered Q&A system backed by **Endee Vector Database**. "
        "Upload PDF/TXT documents and ask questions in natural language."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow local frontend (index.html served from file:// or a dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main findings of the report?",
                "top_k": 5,
            }
        }


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_stored: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list
    top_k_retrieved: int


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """
    Basic health check — confirms the server is running.
    """
    return {
        "status": "ok",
        "service": "AI Knowledge Assistant",
        "version": "1.0.0",
        "vector_db": "Endee",
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    **Upload and index a document.**

    Accepts `.pdf` or `.txt` files.

    Pipeline:
    1. Save file to a temporary location.
    2. Extract and chunk the text.
    3. Generate sentence-transformer embeddings for each chunk.
    4. Upsert vectors and metadata into Endee.

    Returns the number of chunks stored in the vector database.
    """
    # ── Validate extension ────────────────────────────────────────────────────
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # ── Save to temp file ─────────────────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        await file.close()

    # ── Process pipeline ──────────────────────────────────────────────────────
    try:
        logger.info(f"Processing '{filename}' …")

        # Step 1: Extract text and chunk
        chunks = load_and_chunk_document(tmp_path)
        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from the document.")

        logger.info(f"  → {len(chunks)} chunks created")

        # Step 2: Generate embeddings in batch (fast)
        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings_batch(texts)
        logger.info(f"  → {len(embeddings)} embeddings generated")

        # Step 3: Upsert into Endee
        # Attach the original filename as the source identifier
        for chunk in chunks:
            chunk["source"] = filename

        count = upsert_chunks(chunks, embeddings)
        logger.info(f"  → {count} vectors stored in Endee ✓")

        return UploadResponse(
            message=f"Document '{filename}' indexed successfully.",
            filename=filename,
            chunks_stored=count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing '{filename}'")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(payload: QuestionRequest) -> AskResponse:
    """
    **Ask a question about uploaded documents.**

    RAG Pipeline:
    1. Embed the user question via sentence-transformers.
    2. Query Endee for the top-k semantically similar chunks.
    3. Build a context-rich prompt from retrieved chunks.
    4. Send prompt to the LLM (OpenAI / HuggingFace).
    5. Return the answer with source citations.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 characters).")

    try:
        result = run_rag_pipeline(question)
        return AskResponse(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            top_k_retrieved=result["top_k_retrieved"],
        )
    except ValueError as e:
        # Configuration errors (e.g., missing API key) — surface as 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error in /ask endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Admin"])
def index_stats() -> Dict[str, Any]:
    """
    Return Endee index statistics (vector count, dimension, space type).
    """
    stats = get_index_stats()
    return {"endee_index": stats}
