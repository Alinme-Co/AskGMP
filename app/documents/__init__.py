"""
app.documents — read a document of any supported format into records, and
chunk them.

    from app.documents import read_document, extract_records, chunk_records

    records = read_document("spec.xlsx")
    # [{"text": "Parameter: pH | Limit: 6.8-7.2", "page": None,
    #   "section": "Sheet 'Specs', row 4", "atomic": True,
    #   "meta": {"kind": "row", "sheet": "Specs", "row": 4, ...}}, ...]

This package knows about file formats and nothing else. It does not know what a
regulation is, where uploads are stored, or that Qdrant exists — which is what
lets both workflows use it:

    app.knowledge_base  → extract_records() + the cache, over KnowledgeBase/
    app.uploads         → read_document(), over one user's file

**One record model.** Everything produced here has the shape defined in
app/documents/models.py, so a passage from an uploaded SOP and a passage from a
EudraLex annex flow through the same chunker, the same context builder and the
same citation validator with no translation step.

**OCR is applied, not skipped.** For a knowledge base a scanned SOP, a
photographed logbook page or a process flow drawn inside a PowerPoint picture
is content, and ignoring it silently loses it.

**Provenance is structured.** Not just "page 7" but "Sheet 'Batch Record'!C14",
"Slide 4, table 2", "Figure 3 on page 7" — so an answer can point at the cell
or the figure rather than at the file.
"""

from app.core.exceptions import DocumentReadFailed, UnsupportedFormat
from app.documents.chunking import chunk_records, token_count
from app.documents.extractors import (
    OCR_CAPABLE,
    PROFILE_RICH,
    PROFILE_STRUCTURAL,
    SUPPORTED_SUFFIXES,
    UPLOAD_TYPES,
    extract_records,
    format_label,
    is_supported,
    read_document,
    read_summary,
)
from app.documents.models import record, summarise
from app.documents.ocr import ImageOCR, shared_ocr

__all__ = [
    "read_document",
    "read_summary",
    "extract_records",
    "chunk_records",
    "token_count",
    "is_supported",
    "format_label",
    "SUPPORTED_SUFFIXES",
    "UPLOAD_TYPES",
    "OCR_CAPABLE",
    "PROFILE_RICH",
    "PROFILE_STRUCTURAL",
    "UnsupportedFormat",
    "DocumentReadFailed",
    "ImageOCR",
    "shared_ocr",
    "record",
    "summarise",
]
