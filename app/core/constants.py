"""
constants.py — names that must agree across the whole system.

Nothing here is configurable and nothing here is derived from the environment:
these are the field names, discriminators and vector names written into Qdrant
payloads. If the indexer and the retriever ever disagreed about one of them the
symptom would be "retrieval is a bit worse" rather than an error, so they are
defined once and imported everywhere.

Configuration — collection name, URLs, models, limits — lives in app/config.py.
"""

from __future__ import annotations

# ── Named vectors ─────────────────────────────────────────────────────────────
DENSE_VECTOR  = "dense"
SPARSE_VECTOR = "sparse"


# ── Payload discriminators ────────────────────────────────────────────────────
# The collection is shared. `doc_type` separates a European Commission annex
# from a site's own procedure, and collapsing the two would make that
# difference unrecoverable at retrieval time.
DOC_TYPE_FIELD      = "doc_type"
DOC_TYPE_REGULATION = "regulation"   # external regulatory knowledge base
DOC_TYPE_SOP        = "sop"          # a site's own internal procedures
DOC_TYPE_DEVIATION  = "deviation"    # deviation / CAPA records

SOURCE_FIELD  = "source"
SOURCE_ASKGMP = "askgmp"


# ── Where a retrieved passage came from ───────────────────────────────────────
# Written onto an upload's payload as `origin`; absent on knowledge-base points,
# which is what makes ORIGIN_KNOWLEDGE_BASE the default rather than a value that
# has to be backfilled into 1,844 existing points.
ORIGIN_FIELD          = "origin"
ORIGIN_UPLOAD         = "upload"
ORIGIN_KNOWLEDGE_BASE = "knowledge_base"


# ── Payload keys worth a Qdrant index ─────────────────────────────────────────
# The ones a retriever is likely to filter on: "effective EU guidance about
# data integrity", "FDA only", and so on.
INDEXED_KEYWORD_FIELDS = (
    DOC_TYPE_FIELD,
    SOURCE_FIELD,
    "document_id",
    "authority",
    "jurisdiction",
    "source_family",
    "document_type",
    "reference_code",
    "topic",
    "status",
    "regulatory_weight",
    "language",
)


# ── Payload keys copied onto every retrieved chunk ────────────────────────────
# These are the names actually written by the knowledge-base indexer's
# build_payload() — verified against live points rather than assumed.
PAYLOAD_FIELDS = (
    "text",
    "document_id",
    "document_key",
    "artifact_id",
    "document_title",
    "authority",
    "jurisdiction",
    "source_family",
    "document_type",
    "content_role",
    "reference_code",
    "topic",
    "version",
    "publication_date",
    "effective_date",
    "status",
    "regulatory_weight",
    "source_page_url",
    "document_url",
    "citation_url",
    "official_source",
    "section",
    "page",
    "page_end",
    "chunk_no",
    "chunk_id",
    "source_file",
    "source_path",
    "language",
)
