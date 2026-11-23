"""
service.py — store an upload and read it, in one call.

Sits between the UI and the extractors so the frontend never handles paths,
temporary files or extension checks itself, and so an API endpoint can reuse
exactly the same guards.

    document = process_upload(file_bytes, "batch_record.xlsx")
    document.ok        # False if it could not be read
    document.records   # what app.documents made of it
    document.file_hash # the identifier everything downstream uses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import DocumentReadFailed, UnsupportedFormat
from app.documents.extractors import format_label, read_document
from app.documents.models import summarise
from app.uploads.storage import store

logger = logging.getLogger(__name__)


@dataclass
class UploadedDocument:
    """One stored upload and what was read out of it."""

    filename: str                  # as the user named it
    path: Path                     # where it now lives
    suffix: str
    format_label: str
    size_bytes: int
    file_hash: str
    uploaded_at: str
    records: list[dict] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.records)

    @property
    def text(self) -> str:
        """The whole document as one string, for preview or a quick scan."""
        return "\n\n".join(item["text"] for item in self.records if item.get("text"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "format": self.format_label,
            "size_bytes": self.size_bytes,
            "file_hash": self.file_hash,
            "uploaded_at": self.uploaded_at,
            "records": len(self.records),
            "summary": self.summary,
            "error": self.error,
        }


def process_upload(
    data,
    filename: str,
    use_ocr: bool = True,
    upload_dir: Path | None = None,
) -> UploadedDocument:
    """
    Store and read one upload.

    A read failure is returned on the object rather than raised: a batch of
    five files where one is corrupt should report that one and still show the
    other four.
    """
    path, digest, size = store(data, filename, upload_dir=upload_dir)
    suffix = path.suffix.lower()

    document = UploadedDocument(
        filename=filename,
        path=path,
        suffix=suffix,
        format_label=format_label(suffix),
        size_bytes=size,
        file_hash=digest,
        uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    try:
        document.records = read_document(path, use_ocr=use_ocr)
        document.summary = summarise(document.records)
        if not document.records:
            document.error = (
                "No text could be read from this file. If it is a scan, enable "
                "OCR; if it is a form, the content may be in an unsupported layer."
            )
    except (UnsupportedFormat, DocumentReadFailed) as exc:
        document.error = str(exc)
    except Exception as exc:                      # noqa: BLE001
        logger.exception("Unexpected failure reading %s", filename)
        document.error = f"{type(exc).__name__}: {exc}"

    return document


def process_uploads(files, use_ocr: bool = True, upload_dir: Path | None = None):
    """
    Read several uploads, sharing one OCR engine.

    `files` are Streamlit UploadedFile objects, or anything with `.name` and
    `.getvalue()`/`.read()`.
    """
    results: list[UploadedDocument] = []
    for item in files or []:
        name = getattr(item, "name", "document")
        if hasattr(item, "getvalue"):
            data = item.getvalue()
        elif hasattr(item, "read"):
            data = item.read()
        else:
            data = item
        results.append(process_upload(data, name, use_ocr=use_ocr, upload_dir=upload_dir))
    return results
