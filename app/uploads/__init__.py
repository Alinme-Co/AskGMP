"""
app.uploads — the life of a document a user hands to AskGMP.

    storage.py    bytes → data/uploads/, with the guards; hash-addressed
    service.py    store + read, into an UploadedDocument
    search.py     chunk, embed and search it in memory
    summarise.py  what it says, when nobody has asked a question yet

An upload is deliberately *not* written into the shared Qdrant collection. It
has no authority, no regulatory status and no registry row, and it belongs to
one person's question rather than to the corpus. Keeping it in memory avoids a
session filter and a cleanup job whose only purpose would be to undo the
decision to store it in the first place.

What it does share is everything that makes results comparable: the same
extraction stack (app.documents), the same chunker, the same embedding model
and sparse builder (app.infrastructure), and the same evidence type
(app.retrieval.schemas). That is what lets an answer weigh a site's own SOP
beside a EudraLex annex in one response.
"""

from app.uploads.search import UploadIndex, document_id_for, shared_index
from app.uploads.service import UploadedDocument, process_upload, process_uploads
from app.uploads.storage import (
    MAX_UPLOAD_BYTES,
    UPLOAD_DIR,
    content_hash,
    find_by_hash,
    original_stem,
    safe_name,
    store,
)

__all__ = [
    "UploadIndex",
    "shared_index",
    "document_id_for",
    "UploadedDocument",
    "process_upload",
    "process_uploads",
    "store",
    "safe_name",
    "content_hash",
    "find_by_hash",
    "original_stem",
    "UPLOAD_DIR",
    "MAX_UPLOAD_BYTES",
]
