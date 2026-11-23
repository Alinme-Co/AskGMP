"""
search.py — RAG over an uploaded document, in memory.

An uploaded SOP is not part of the knowledge base and should not be. It has no
authority, no `status`, no registry row, and it belongs to one person's
question rather than to the corpus — so writing it into the shared Qdrant
collection would mean a filter, a session key and a cleanup job to keep one
user's draft procedure out of everyone else's search results.

Instead the document is chunked with the project's own chunker, embedded once,
and searched in memory:

    records → chunk_records() → embed once (cached by file hash)
      → dense cosine + sparse dot → RRF fusion → best chunks

The retrieval mirrors the knowledge base's deliberately. Same extraction stack
(app.documents), same chunker, same embedding model, same crc32 sparse builder,
same RRF constant — so a passage found in an uploaded SOP was found the same
way as a passage found in EudraLex, and the two can be ranked beside each other
honestly. The sparse half matters more here than it does for the corpus: an
internal SOP is full of identifiers ("ZYNTAROL-40", "SOP-9931", a batch number)
that a dense embedding smooths away and a keyword match finds exactly.

Results come back as `RetrievedChunk`, the same evidence type the Qdrant
retriever returns, so every downstream step — context building, citation
validation, source-table aggregation — works on uploaded chunks with no
special case.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from app import config
from app.core.constants import ORIGIN_FIELD, ORIGIN_UPLOAD
from app.documents.chunking import chunk_records
from app.documents.extractors import read_document
from app.infrastructure.embeddings import sparse_map
from app.infrastructure.openai_client import get_openai_client
from app.retrieval.ranking import reciprocal_rank_fusion
from app.retrieval.schemas import RetrievedChunk
from app.uploads.storage import find_by_hash, original_stem

logger = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def document_id_for(filename: str, digest: str) -> str:
    """
    A citation-friendly, collision-proof id for an uploaded document.

    Prefixed UPLOAD_ so the identifier itself says where the text came from:
    seeing "[UPLOAD_SOP_9931, p. 2]" beside "[EU_GMP_CHAPTER_1, p. 8]" in an
    answer tells the reader which claim rests on their own procedure and which
    on the regulation, without having to consult the table.
    """
    slug = _NON_ALNUM.sub("_", original_stem(filename)).strip("_").upper()[:40]
    return f"UPLOAD_{slug or 'DOCUMENT'}_{digest[:6].upper()}"


@dataclass
class IndexedDocument:
    """One uploaded document, chunked and embedded."""

    file_hash: str
    filename: str
    document_id: str
    path: Path | None = None
    chunks: list[dict] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    sparse: list[dict[int, float]] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class UploadIndex:
    """
    Chunk, embed and search uploaded documents.

    Embeddings are cached by file hash for the life of the process. Uploading
    the same document twice, or asking it five questions, embeds it once.
    """

    def __init__(self, openai_client=None):
        self._openai = openai_client
        self._documents: dict[str, IndexedDocument] = {}

    @property
    def openai(self):
        if self._openai is None:
            self._openai = get_openai_client()
        return self._openai

    # ── Building ──────────────────────────────────────────────────────────────
    def add_records(
        self,
        records: list[dict],
        filename: str,
        file_hash: str,
        path: Path | None = None,
    ) -> IndexedDocument:
        """Chunk and embed already-read records."""
        if file_hash in self._documents:
            return self._documents[file_hash]

        chunks = chunk_records(records)
        document = IndexedDocument(
            file_hash=file_hash,
            filename=filename,
            document_id=document_id_for(filename, file_hash),
            path=path,
            chunks=chunks,
        )

        if chunks:
            texts = [chunk["text"] for chunk in chunks]
            document.vectors = self._embed(texts)
            document.sparse = [sparse_map(text) for text in texts]

        self._documents[file_hash] = document
        return document

    def add_path(self, path: str | Path, file_hash: str, use_ocr: bool = True) -> IndexedDocument:
        """Read a stored upload from disk, then chunk and embed it."""
        if file_hash in self._documents:
            return self._documents[file_hash]

        path = Path(path)
        records = read_document(path, use_ocr=use_ocr)
        return self.add_records(records, path.name, file_hash, path=path)

    def ensure(self, file_hash: str, use_ocr: bool = True) -> IndexedDocument | None:
        """
        The indexed document for a hash, reading it from data/uploads/ if this
        process has not seen it yet.
        """
        if file_hash in self._documents:
            return self._documents[file_hash]

        path = find_by_hash(file_hash)
        if path is None:
            logger.warning("No stored upload for hash %s", file_hash[:12])
            return None

        try:
            return self.add_path(path, file_hash, use_ocr=use_ocr)
        except Exception as exc:                  # noqa: BLE001
            logger.warning("Could not index %s: %s", path.name, exc)
            return None

    def get(self, file_hash: str) -> IndexedDocument | None:
        return self._documents.get(file_hash)

    def _embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.openai.embeddings.create(
                model=config.EMBED_MODEL, input=batch
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    # ── Searching ─────────────────────────────────────────────────────────────
    def search(
        self,
        question: str,
        file_hashes: list[str],
        limit: int | None = None,
        use_ocr: bool = True,
    ) -> list[RetrievedChunk]:
        """
        The passages of the uploaded documents that bear on the question.

        Searched across every selected document at once and ranked together, so
        two uploaded SOPs compete on relevance rather than the first one
        winning by position.
        """
        question = (question or "").strip()
        limit = limit or config.MAX_DOCUMENT_CHUNKS
        if not question or not file_hashes or limit <= 0:
            return []

        documents = [
            document
            for document in (self.ensure(h, use_ocr=use_ocr) for h in file_hashes)
            if document is not None and document.chunk_count
        ]
        if not documents:
            return []

        query_vector = self._embed([question])[0]
        query_sparse = sparse_map(question)

        dense_pairs: list[tuple[float, tuple[IndexedDocument, int]]] = []
        sparse_pairs: list[tuple[float, tuple[IndexedDocument, int]]] = []

        for document in documents:
            for position, chunk in enumerate(document.chunks):
                if position < len(document.vectors):
                    dense_pairs.append((
                        _cosine(query_vector, document.vectors[position]),
                        (document, position),
                    ))
                if position < len(document.sparse):
                    sparse_pairs.append((
                        _sparse_dot(query_sparse, document.sparse[position]),
                        (document, position),
                    ))

        fused = reciprocal_rank_fusion(dense_pairs, sparse_pairs)

        results: list[RetrievedChunk] = []
        for score, (document, position) in fused[:limit]:
            results.append(_to_retrieved(document, position, score))

        return results

    def all_chunks(self, file_hash: str) -> list[RetrievedChunk]:
        """Every chunk of one document, in order — used for summarising."""
        document = self.ensure(file_hash)
        if document is None:
            return []
        return [
            _to_retrieved(document, position, 1.0)
            for position in range(document.chunk_count)
        ]


# ── Scoring helpers ───────────────────────────────────────────────────────────
def _sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller side; a query has far fewer terms than a chunk.
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(index, 0.0) for index, value in a.items())


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _to_retrieved(document: IndexedDocument, position: int, score: float) -> RetrievedChunk:
    """
    One uploaded chunk as the same type the Qdrant retriever returns.

    The payload deliberately mirrors an indexed chunk's, with the regulatory
    fields left empty rather than invented: an uploaded document has no
    authority, no reference code and no official URL, and filling those in
    would make a site's own draft look like published guidance.
    """
    chunk = document.chunks[position]
    meta = chunk.get("meta") or {}

    payload = {
        "text": chunk["text"],
        "document_id": document.document_id,
        "document_key": document.file_hash,
        "document_title": original_stem(document.filename).replace("_", " "),
        "authority": None,
        "jurisdiction": None,
        "source_family": None,
        "document_type": "uploaded document",
        "content_role": meta.get("kind"),
        "reference_code": None,
        "topic": None,
        "status": None,
        "regulatory_weight": None,
        "source_page_url": None,
        "document_url": None,
        "citation_url": None,
        "official_source": False,
        "section": chunk.get("section"),
        "page": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "chunk_no": position,
        "chunk_id": f"{document.document_id}#chunk-{position:04d}",
        "source_file": document.filename,
        "source_path": str(document.path) if document.path else None,
        "language": None,
        ORIGIN_FIELD: ORIGIN_UPLOAD,
        "locator": meta.get("locator"),
    }

    return RetrievedChunk(text=chunk["text"], score=float(score), payload=payload)


_SHARED: UploadIndex | None = None


def shared_index() -> UploadIndex:
    """One index per process, so embeddings are reused across questions."""
    global _SHARED
    if _SHARED is None:
        _SHARED = UploadIndex()
    return _SHARED
