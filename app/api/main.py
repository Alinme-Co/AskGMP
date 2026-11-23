"""
main.py — the AskGMP HTTP API.

    POST /api/chat     {"question": "..."} -> {"answer": ..., "sources": [...]}
    GET  /api/health   readiness, including whether Qdrant is actually reachable
    GET  /api/config   the retrieval settings currently in force

The API returns JSON and nothing else. Building the source *table* is the
frontend's job — the backend's job is to return the rows. Keeping a pandas
DataFrame out of the response is what lets the same endpoint serve the
Streamlit app, a React frontend, or curl.

    uvicorn app.api.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.chat.schemas import (
    AskGMPResponse,
    ChatRequest,
    DocumentSummary,
    SummaryRequest,
)
from app.chat.service import AskGMPService
from app.uploads.summarise import DocumentSummariser

app = FastAPI(
    title="AskGMP",
    version="1.0.0",
    description="Grounded question answering over the AskGMP regulatory knowledge base.",
)

# Wide open, which is fine for a local V1 and must not stay that way if this is
# ever exposed beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_service: AskGMPService | None = None
_summariser: DocumentSummariser | None = None


def get_service() -> AskGMPService:
    """
    One service for the process.

    Built lazily rather than at import time so the module can be imported —
    by the tests, or by `--help` — without needing a Qdrant connection or an
    API key.
    """
    global _service
    if _service is None:
        _service = AskGMPService()
    return _service


@app.post("/api/chat", response_model=AskGMPResponse)
def chat(request: ChatRequest) -> AskGMPResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        return get_service().answer(
            question,
            effective_only=request.effective_only,
            authority=request.authority,
            document_hashes=request.document_hashes,
        )
    except HTTPException:
        raise
    except Exception as exc:                     # noqa: BLE001
        # Surface the failure rather than returning an empty answer, which the
        # frontend would render as a legitimate "nothing found".
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc


def get_summariser() -> DocumentSummariser:
    global _summariser
    if _summariser is None:
        _summariser = DocumentSummariser(upload_index=get_service().uploads)
    return _summariser


@app.post("/api/documents/summary", response_model=DocumentSummary)
def summarise_document(request: SummaryRequest) -> DocumentSummary:
    """
    What an uploaded document says, for when no question has been asked.

    Takes a hash rather than the file: `store` already wrote the document to
    data/uploads/ under a hash-bearing name, so the browser holds an identifier
    and the bytes never travel twice.
    """
    try:
        return get_summariser().summarise(request.file_hash, request.filename)
    except Exception as exc:                     # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"{type(exc).__name__}: {exc}"
        ) from exc


@app.get("/api/health")
def health() -> dict:
    from app.infrastructure.qdrant import get_client

    collection = config.COLLECTION_NAME
    status = {"status": "ok", "collection": collection, "qdrant": "unknown", "points": None}

    try:
        client = get_client()
        info = client.get_collection(collection)
        status["qdrant"] = "connected"
        status["points"] = info.points_count
        core = get_service().core
        if core is not None:
            from app.chat import service

            status["cag_core_chunks"] = len(core.chunks())
            # Real provider numbers from the last answer; None until one is asked.
            status["cag_last_usage"] = service.last_usage
    except Exception as exc:                     # noqa: BLE001
        status["status"] = "degraded"
        status["qdrant"] = f"unavailable: {type(exc).__name__}"

    return status


@app.get("/api/config")
def settings() -> dict:
    return {
        "chat_model": config.CHAT_MODEL,
        "candidate_limit": config.CANDIDATE_LIMIT,
        "max_context_chunks": config.MAX_CONTEXT_CHUNKS,
        "max_chunks_per_document": config.MAX_CHUNKS_PER_DOCUMENT,
        "effective_only": config.EFFECTIVE_ONLY,
        "cag_enabled": config.CAG_ENABLED,
        "cag_core_documents": config.CAG_CORE_DOCUMENTS,
    }
