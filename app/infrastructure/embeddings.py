"""
embeddings.py — both halves of the hybrid vector, built the same way everywhere.

Dense vectors come from OpenAI. Sparse vectors are a crc32 term-frequency hash,
built here and nowhere else — a second, separately-fitted vocabulary (a TF-IDF
vectorizer, say) would produce term IDs in a different space and silently break
the keyword half of every search, in a way that looks like "retrieval is a bit
worse" rather than like a bug.

Three callers depend on that agreement: the knowledge-base indexer writing
points, the retriever querying them, and the in-memory index over user uploads.
"""

from __future__ import annotations

import math
import re
import time
import zlib
from collections import Counter

from app import config

# ── Sparse vectors ────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-/.]+", re.IGNORECASE)

# A short list only. Regulatory prose leans hard on "shall"/"should"/"must",
# and those words carry real meaning here, so they are not stopped.
_SPARSE_STOPWORDS = frozenset(
    """
    a an the and or of to in for on at by with from as is are was were be been
    this that these those it its their there which who whom whose into than then
    """.split()
)

_SPARSE_MODULUS = 1 << 31


def build_sparse_terms(text: str) -> tuple[list[int], list[float]]:
    """
    (indices, values) for the sparse half of hybrid search.

    Log-damped term frequency, L2-normalised, with term IDs from crc32 of the
    token. Hashing rather than a fitted vocabulary is what lets the query side
    produce compatible vectors without loading any state.
    """
    counts = Counter(
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        if token not in _SPARSE_STOPWORDS
    )
    if not counts:
        return [], []

    indices: list[int] = []
    values: list[float] = []
    for term, frequency in counts.items():
        indices.append(zlib.crc32(term.encode("utf-8")) % _SPARSE_MODULUS)
        values.append(1.0 + math.log(frequency))

    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return indices, [value / norm for value in values]


def build_sparse_vector(text: str):
    """The same thing as a qdrant SparseVector. Needs qdrant_client."""
    from qdrant_client.models import SparseVector

    indices, values = build_sparse_terms(text)
    return SparseVector(indices=indices, values=values)


def sparse_map(text: str) -> dict[int, float]:
    """The same sparse vector as a dict, for a cheap in-memory dot product."""
    indices, values = build_sparse_terms(text)
    return dict(zip(indices, values))


# ── Dense vectors ─────────────────────────────────────────────────────────────
def embed_texts(
    openai_client,
    model: str | None = None,
    texts: list[str] | None = None,
    batch_size: int = 20,
    retries: int = 3,
    sleep: float = 0.3,
    label: str = "chunks",
) -> list[list[float]]:
    """
    Embed in batches, retrying each batch with a widening backoff.

    A rate limit or a dropped connection twenty minutes into a corpus-sized run
    must not throw the work away, so a failed batch is retried rather than
    raised. After `retries` it does raise — silently returning short vectors
    would corrupt the index.
    """
    model = model or config.EMBED_MODEL
    texts = texts or []
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        for attempt in range(1, retries + 1):
            try:
                response = openai_client.embeddings.create(model=model, input=batch)
                vectors.extend(item.embedding for item in response.data)
                print(f"    embedded {len(vectors)}/{len(texts)} {label}", end="\r")
                time.sleep(sleep)
                break
            except Exception as exc:
                if attempt == retries:
                    raise RuntimeError(
                        f"Embedding failed after {retries} attempts on the batch "
                        f"starting at index {start}: {exc}"
                    ) from exc
                print(f"\n    retry {attempt}/{retries} for embed batch at {start}: {exc}")
                time.sleep(sleep * attempt * 4)

    print()
    return vectors
