"""
ranking.py — what happens to candidates after they come back.

Two functions, both about the same problem: a ranked list is not yet a good
context window.

`diversify` caps how many chunks one document may contribute, and is used by
the Qdrant retriever. `reciprocal_rank_fusion` fuses two rankings the way
Qdrant's own RRF does, and is used by the in-memory index over uploads — where
there is no Qdrant to do it. Keeping them side by side is deliberate: if the
constant `k` here ever drifted from Qdrant's, scores from the two halves would
stop being comparable and nobody would see an error.
"""

from __future__ import annotations

from typing import Any

from app.retrieval.schemas import RetrievedChunk

#: The constant Qdrant's RRF implementation uses. Kept the same so an in-memory
#: fused score and a Qdrant fused score mean the same thing.
RRF_K = 60


def diversify(
    chunks: list[RetrievedChunk],
    max_chunks: int,
    max_per_document: int,
) -> list[RetrievedChunk]:
    """
    Keep the best chunks while limiting how many any one document contributes.

    A question like "what does guidance say about data integrity?" matches the
    PIC/S PI 041-1 document on almost every page, so an undiversified top-8 is
    eight chunks of one document and the answer cannot mention that WHO, MHRA
    and the EMA Q&A also address it. Capping per document costs a little depth
    and buys the cross-authority comparison the tool exists for.

    Two passes on purpose. The first respects the per-document cap, so several
    documents are represented even when one dominates the ranking. The second
    backfills from what the cap excluded, so a question that genuinely has a
    single relevant document still gets a full context window instead of one
    lonely chunk.
    """
    if max_chunks <= 0:
        return []

    selected: list[RetrievedChunk] = []
    per_document: dict[str, int] = {}
    overflow: list[RetrievedChunk] = []

    for chunk in chunks:
        key = chunk.document_id or chunk.get("document_key") or ""
        if per_document.get(key, 0) >= max_per_document:
            overflow.append(chunk)
            continue
        per_document[key] = per_document.get(key, 0) + 1
        selected.append(chunk)
        if len(selected) >= max_chunks:
            return selected

    for chunk in overflow:
        if len(selected) >= max_chunks:
            break
        selected.append(chunk)

    return selected


def reciprocal_rank_fusion(dense_pairs, sparse_pairs, k: int = RRF_K):
    """
    Fuse two rankings of (score, holder) pairs the way Qdrant does.

    RRF uses rank rather than score, which is the point: a cosine similarity
    and a term-frequency dot product are not on the same scale and adding them
    would let whichever happens to be larger dominate.
    """
    scores: dict[tuple[int, int], float] = {}
    holders: dict[tuple[int, int], Any] = {}

    for pairs in (dense_pairs, sparse_pairs):
        ranked = sorted(pairs, key=lambda pair: -pair[0])
        for rank, (score, holder) in enumerate(ranked):
            if score <= 0.0:
                continue
            document, position = holder
            key = (id(document), position)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            holders[key] = holder

    fused = [(score, holders[key]) for key, score in scores.items()]
    fused.sort(key=lambda pair: -pair[0])
    return fused
