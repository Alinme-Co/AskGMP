"""
qdrant.py — the Qdrant collection contract, owned by nobody in particular.

Everything that has to agree between the indexer and the retriever lives here:
client construction, collection creation, the deterministic point-ID scheme,
point construction with both named vectors, and filter building.

This module is deliberately not owned by the knowledge-base package. The
indexer writes points and the retriever reads them; if either owned the
contract the other would have to import it, and the dependency arrow would run
sideways between two peers instead of down into shared infrastructure.

qdrant_client is imported inside the functions that need it, so payload
construction, chunking and the whole --dry-run path can run — and be unit
tested — without the package installed.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from app import config
from app.core.constants import (
    DENSE_VECTOR,
    INDEXED_KEYWORD_FIELDS,
    SPARSE_VECTOR,
)
from app.infrastructure.embeddings import build_sparse_vector

# ── Point IDs ─────────────────────────────────────────────────────────────────
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def point_id(doc_type: str, chunk_id: str) -> str:
    """
    Deterministic point ID.

    Deterministic so that re-indexing a changed document overwrites its previous
    points in place instead of accumulating duplicates alongside them.
    """
    return str(uuid.uuid5(_NAMESPACE, f"{doc_type}:{chunk_id}"))


# ── Client / collection ───────────────────────────────────────────────────────
def get_client():
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=config.QDRANT_TIMEOUT,
    )


def ensure_collection(client, dense_dim: int | None = None,
                      collection: str | None = None) -> None:
    """Create the collection and its payload indexes if they are not there yet."""
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        SparseVectorParams,
        VectorParams,
    )

    collection = collection or config.COLLECTION_NAME
    dense_dim = dense_dim or config.DENSE_DIM

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name       = collection,
            vectors_config        = {
                DENSE_VECTOR: VectorParams(size=dense_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config = {SPARSE_VECTOR: SparseVectorParams()},
        )
        print(f"  Created collection '{collection}' ({dense_dim}d dense + sparse).")
    else:
        print(f"  Using existing collection '{collection}'.")

    for field in INDEXED_KEYWORD_FIELDS:
        try:
            client.create_payload_index(
                collection_name = collection,
                field_name      = field,
                field_schema    = PayloadSchemaType.KEYWORD,
            )
        except Exception:            # noqa: BLE001 - "already indexed" is the norm
            pass


def make_point(pid: str, dense: list[float], text: str, payload: dict):
    """One PointStruct carrying both named vectors."""
    from qdrant_client.models import PointStruct

    return PointStruct(
        id      = pid,
        vector  = {
            DENSE_VECTOR : dense,
            SPARSE_VECTOR: build_sparse_vector(text),
        },
        payload = payload,
    )


# ── Filters ───────────────────────────────────────────────────────────────────
def build_filter(conditions: dict):
    """A must-match-all Filter from a plain {field: value} mapping."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in conditions.items()
    ])


# ── Deletion / counting ───────────────────────────────────────────────────────
def delete_by_filter(client, conditions: dict, collection: str | None = None) -> None:
    from qdrant_client.models import FilterSelector

    client.delete(
        collection_name = collection or config.COLLECTION_NAME,
        points_selector = FilterSelector(filter=build_filter(conditions)),
        wait            = True,
    )


def delete_point_ids(client, ids: list[str], batch_size: int = 256,
                     collection: str | None = None) -> int:
    from qdrant_client.models import PointIdsList

    collection = collection or config.COLLECTION_NAME
    deleted = 0
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        client.delete(
            collection_name = collection,
            points_selector = PointIdsList(points=batch),
            wait            = True,
        )
        deleted += len(batch)
    return deleted


def count_points(client, conditions: dict, collection: str | None = None) -> int:
    try:
        return client.count(
            collection_name = collection or config.COLLECTION_NAME,
            count_filter    = build_filter(conditions),
            exact           = True,
        ).count
    except Exception:                # noqa: BLE001 - reporting only
        return -1


# ── Upload with retries ───────────────────────────────────────────────────────
def upsert_in_batches(
    client,
    collection: str,
    points: list,
    batch_size: int = 10,
    retries: int = 3,
    sleep: float = 0.5,
    on_batch: Callable[[list], None] | None = None,
) -> int:
    """
    Upsert points in batches, retrying each batch with a widening backoff.

    `on_batch` runs after a batch is durably written — the indexer uses it to
    checkpoint crash-recovery state, so progress recorded on disk never claims
    more than Qdrant actually holds.
    """
    uploaded = 0

    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]

        for attempt in range(1, retries + 1):
            try:
                client.upsert(collection_name=collection, points=batch, wait=True)
                uploaded += len(batch)
                if on_batch:
                    on_batch(batch)
                print(f"    uploaded {uploaded}/{len(points)} points", end="\r")
                time.sleep(sleep)
                break
            except Exception as exc:
                if attempt == retries:
                    raise RuntimeError(
                        f"Upload failed after {retries} attempts on the batch "
                        f"starting at index {start}: {exc}"
                    ) from exc
                print(f"\n    retry {attempt}/{retries} for batch at {start}: {exc}")
                time.sleep(sleep * attempt * 2)

    print()
    return uploaded
