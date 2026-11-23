"""
retriever.py — Hybrid search over the AskGMP knowledge base.

Dense and sparse queries run as two Qdrant prefetches and are fused with
Reciprocal Rank Fusion. Both halves come from app/infrastructure/embeddings.py:
the same embedding model that produced the stored dense vectors, and the same
crc32 term-frequency builder that produced the stored sparse ones. That reuse
is not incidental — a separately fitted sparse vocabulary would put term IDs in
a different space and quietly break the keyword half of every search, in a way
that looks like "retrieval is a bit worse" rather than like a bug.

The pipeline:

    embed question
      → dense prefetch + sparse prefetch (wide, PREFETCH_LIMIT each)
        → RRF fusion (CANDIDATE_LIMIT)
          → drop non-effective documents (unless asked for)
            → diversify: cap chunks per document
              → MAX_CONTEXT_CHUNKS best chunks

This module depends on app.infrastructure, never on app.knowledge_base. The
indexer writes the points and the retriever reads them, but neither imports the
other: what they agree on is the collection contract, and that lives in
app/core/constants.py and app/infrastructure/qdrant.py.
"""

from __future__ import annotations

from app import config
from app.core.constants import DENSE_VECTOR, PAYLOAD_FIELDS, SPARSE_VECTOR
from app.infrastructure.embeddings import build_sparse_vector
from app.infrastructure.openai_client import get_openai_client
from app.infrastructure.qdrant import get_client
from app.retrieval.filters import knowledge_base_filter
from app.retrieval.ranking import diversify
from app.retrieval.schemas import RetrievedChunk, RetrievedEvidence

__all__ = ["Retriever", "RetrievedChunk", "RetrievedEvidence", "diversify",
           "PAYLOAD_FIELDS"]


class Retriever:
    """
    Hybrid search against the indexed knowledge base.

    The OpenAI and Qdrant clients are injectable so tests can drive the whole
    pipeline — filtering, diversification, ranking — without a network call.
    """

    def __init__(self, qdrant_client=None, openai_client=None, collection: str | None = None):
        self._qdrant = qdrant_client
        self._openai = openai_client
        self.collection = collection or config.COLLECTION_NAME

    # -- lazily built clients, so importing this module costs nothing --
    @property
    def qdrant(self):
        if self._qdrant is None:
            self._qdrant = get_client()
        return self._qdrant

    @property
    def openai(self):
        if self._openai is None:
            self._openai = get_openai_client()
        return self._openai

    # -- query --
    def embed(self, question: str) -> list[float]:
        response = self.openai.embeddings.create(
            model=config.EMBED_MODEL, input=[question]
        )
        return response.data[0].embedding

    def search(
        self,
        question: str,
        limit: int | None = None,
        effective_only: bool | None = None,
        authority: str | None = None,
    ) -> list[RetrievedChunk]:
        """Fused dense+sparse candidates, filtered and ranked but not yet trimmed."""
        from qdrant_client.models import Fusion, FusionQuery, Prefetch

        question = (question or "").strip()
        if not question:
            return []

        limit = limit or config.CANDIDATE_LIMIT
        query_filter = knowledge_base_filter(
            effective_only=effective_only, authority=authority
        )

        # The filter is applied to both prefetches rather than after fusion:
        # filtering afterwards would let superseded chunks consume candidate
        # slots and quietly shrink the effective result set.
        points = self.qdrant.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=self.embed(question),
                    using=DENSE_VECTOR,
                    limit=config.PREFETCH_LIMIT,
                    filter=query_filter,
                ),
                Prefetch(
                    query=build_sparse_vector(question),
                    using=SPARSE_VECTOR,
                    limit=config.PREFETCH_LIMIT,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

        chunks = [
            RetrievedChunk(
                text=str(point.payload.get("text") or ""),
                score=float(point.score or 0.0),
                payload={key: point.payload.get(key) for key in PAYLOAD_FIELDS},
            )
            for point in points
            if point.payload
        ]

        return [
            chunk for chunk in chunks
            if chunk.text.strip() and chunk.score >= config.MIN_SCORE
        ]

    def retrieve(
        self,
        question: str,
        effective_only: bool | None = None,
        authority: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search, then diversify down to the chunks that go in the prompt."""
        return diversify(
            self.search(question, effective_only=effective_only, authority=authority),
            max_chunks=config.MAX_CONTEXT_CHUNKS,
            max_per_document=config.MAX_CHUNKS_PER_DOCUMENT,
        )
