"""
filters.py — which points a query is allowed to see.

Three conditions, and the third is the one that matters clinically. The
collection is shared, so `doc_type` and `source` keep a site's own SOPs and
deviation records out of a question about what the regulations require. And
`status` keeps superseded and withdrawn guidance out by default — it is still
in the index, because the registry records regulatory state precisely so it can
be filtered rather than deleted, but answering from a withdrawn guideline is a
real hazard rather than a slightly worse result.
"""

from __future__ import annotations

from app import config
from app.core.constants import (
    DOC_TYPE_FIELD,
    DOC_TYPE_REGULATION,
    SOURCE_ASKGMP,
    SOURCE_FIELD,
)


def knowledge_base_filter(
    effective_only: bool | None = None,
    authority: str | None = None,
    doc_type: str = DOC_TYPE_REGULATION,
    source: str = SOURCE_ASKGMP,
):
    """A Qdrant Filter for the regulatory knowledge base."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    effective_only = config.EFFECTIVE_ONLY if effective_only is None else effective_only

    must = [
        FieldCondition(key=DOC_TYPE_FIELD, match=MatchValue(value=doc_type)),
        FieldCondition(key=SOURCE_FIELD, match=MatchValue(value=source)),
    ]
    if effective_only:
        must.append(
            FieldCondition(key="status", match=MatchValue(value=config.EFFECTIVE_STATUS))
        )
    if authority:
        must.append(FieldCondition(key="authority", match=MatchValue(value=authority)))

    return Filter(must=must)
