"""
core.py — the documents every question sees, in full, from a cached prefix.

The core set is read from the same Qdrant collection the retriever searches,
not re-extracted from the source files. That keeps one text and one payload per
chunk: a core citation is validated, rendered and aggregated exactly like a
retrieved one, because it is the same object.

The chunks are loaded once per process and sorted into a fixed order. The
order matters more than it looks: the prompt-cache prefix only hits when the
rendered core block is byte-identical between calls, so anything that could
reorder it — scroll pagination, a set — must not reach the prompt.
"""

from __future__ import annotations

from app import config
from app.core.constants import PAYLOAD_FIELDS
from app.infrastructure.qdrant import get_client
from app.retrieval.filters import knowledge_base_filter
from app.retrieval.schemas import RetrievedChunk


class CoreLibrary:
    """The core documents as chunks, loaded lazily and kept for the process."""

    def __init__(self, qdrant_client=None, document_ids: list[str] | None = None,
                 collection: str | None = None):
        self._qdrant = qdrant_client
        self.document_ids = list(document_ids if document_ids is not None
                                 else config.CAG_CORE_DOCUMENTS)
        self.collection = collection or config.COLLECTION_NAME
        self._chunks: list[RetrievedChunk] | None = None

    @property
    def qdrant(self):
        if self._qdrant is None:
            self._qdrant = get_client()
        return self._qdrant

    def chunks(self) -> list[RetrievedChunk]:
        if self._chunks is None:
            self._chunks = self._load()
        return self._chunks

    def refresh(self) -> None:
        """Drop the loaded chunks, e.g. after a re-index changed a core document."""
        self._chunks = None

    def _load(self) -> list[RetrievedChunk]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        if not self.document_ids:
            return []

        query_filter = Filter(must=[
            knowledge_base_filter(effective_only=True),
            FieldCondition(key="document_id", match=MatchAny(any=self.document_ids)),
        ])

        points, offset = [], None
        while True:
            batch, offset = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                break

        order = {document_id: n for n, document_id in enumerate(self.document_ids)}
        chunks = [
            RetrievedChunk(
                text=str(point.payload.get("text") or ""),
                score=0.0,
                payload={key: point.payload.get(key) for key in PAYLOAD_FIELDS},
            )
            for point in points
            if point.payload and str(point.payload.get("text") or "").strip()
        ]
        # Configured document order, then reading order inside each document.
        chunks.sort(key=lambda c: (order.get(c.document_id, len(order)),
                                   int(c.payload.get("chunk_no") or 0)))
        return chunks
