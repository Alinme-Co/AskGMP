"""
schemas.py — the evidence type everything downstream operates on.

One passage, wherever it came from. A chunk retrieved from the regulatory
Qdrant collection and a chunk found in the user's own uploaded SOP arrive as
the same type, carrying the same field names, so context building, citation
validation and source-table aggregation have no special case for either.

That is what makes the two searchable together. When someone uploads their
deviation procedure and asks "is this situation a deviation?", the answer needs
both what their SOP says and what the regulation requires; if the two halves
had different shapes, every step after retrieval would need to know which was
which.

The regulatory fields — authority, reference_code, status, citation_url — are
present on an upload's payload but left empty rather than invented. A site's
draft procedure must never look like published guidance, and `source_type` is
how a caller tells them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.constants import ORIGIN_FIELD, ORIGIN_KNOWLEDGE_BASE, ORIGIN_UPLOAD


@dataclass
class RetrievedChunk:
    """One chunk plus everything a citation or a source row could need."""

    text: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    # ── Identity ──────────────────────────────────────────────────────────────
    @property
    def document_id(self) -> str:
        return str(self.payload.get("document_id") or "")

    @property
    def document_title(self) -> str:
        return str(self.payload.get("document_title") or "")

    @property
    def source_type(self) -> str:
        """
        "user_upload" or "knowledge_base".

        Derived from the payload rather than stored as a second field, because
        the 1,844 points already in Qdrant carry no `origin` and backfilling
        them would mean rewriting the collection to record something that is
        true of every one of them.
        """
        origin = self.payload.get(ORIGIN_FIELD)
        return "user_upload" if origin == ORIGIN_UPLOAD else "knowledge_base"

    @property
    def is_upload(self) -> bool:
        return self.payload.get(ORIGIN_FIELD) == ORIGIN_UPLOAD

    # ── Location ──────────────────────────────────────────────────────────────
    @property
    def page(self) -> int | None:
        return _as_int(self.payload.get("page"))

    @property
    def page_end(self) -> int | None:
        return _as_int(self.payload.get("page_end"))

    @property
    def section(self) -> str | None:
        value = self.payload.get("section")
        return str(value) if value else None

    # ── Regulatory state ──────────────────────────────────────────────────────
    @property
    def authority(self) -> str | None:
        value = self.payload.get("authority")
        return str(value) if value else None

    @property
    def reference_code(self) -> str | None:
        value = self.payload.get("reference_code")
        return str(value) if value else None

    @property
    def citation_url(self) -> str | None:
        value = self.payload.get("citation_url")
        return str(value) if value else None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


#: The evidence abstraction under its intent-revealing name. `RetrievedChunk`
#: is what the code has always called it and what the tests import; this alias
#: is for call sites that care that the passage may not be from the corpus.
RetrievedEvidence = RetrievedChunk

__all__ = [
    "RetrievedChunk",
    "RetrievedEvidence",
    "ORIGIN_UPLOAD",
    "ORIGIN_KNOWLEDGE_BASE",
]
