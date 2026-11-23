"""
storage.py — take a file from the user and put it somewhere safe.

The guards are the point of this module rather than an afterthought:

* the extension must be one an extractor claims — checked before anything is
  written, so an unsupported file never reaches disk;
* the filename is reduced to a safe stem, because an uploaded name is user
  input and "../../.env" is a valid one;
* a size ceiling, because OCR on a 200 MB scan will hold a request open for
  many minutes;
* a content hash, so the same document uploaded twice is recognised as the
  same document — the knowledge-base extraction cache is keyed the same way.

Uploads land in data/uploads/ alongside the other working data, not in a system
temp folder, so a file survives long enough to be indexed in a later step. The
stored name carries the hash, which is what lets the API stay stateless: the
browser holds an identifier and the backend finds the document from it without
a session table.

Storage only. Reading the file is app/uploads/service.py's job, and what the
formats mean is app/documents'.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import BinaryIO

from app import config
from app.core.exceptions import UnsupportedFormat, UploadRejected
from app.documents.extractors import SUPPORTED_SUFFIXES

UPLOAD_DIR = config.UPLOAD_DIR
MAX_UPLOAD_BYTES = config.MAX_UPLOAD_BYTES

_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")

# `store` names files "<stem>-<first 12 of sha256><suffix>". When a document is
# loaded back off disk the stored name is all we have, so the suffix has to be
# strippable again — otherwise the hash lands in the citation key twice, as
# "UPLOAD_SOP_9931_F29405BEFB4F_F29405".
_STORED_SUFFIX = re.compile(r"-[0-9a-f]{12}$", re.IGNORECASE)


def content_hash(data: bytes) -> str:
    """
    SHA-256 of the bytes, the same identity `store` uses.

    Exposed so a caller can ask "have I already read this?" before paying for
    the read. Streamlit hands back the same uploaded file on every rerun, and
    OCR-ing a scanned document once per keystroke would make the page unusable.
    """
    return hashlib.sha256(data).hexdigest()


def safe_name(filename: str) -> str:
    """
    A filename reduced to something safe to join onto a directory.

    An uploaded name is user input. Taking only the final component and
    stripping everything that is not alphanumeric keeps "../../.env" from being
    a working path.
    """
    stem = Path(filename or "document").name
    cleaned = _SAFE_STEM.sub("_", stem).strip("._") or "document"
    return cleaned[:120]


def original_stem(filename: str) -> str:
    """The user's own filename stem, with any storage hash suffix removed."""
    stem = Path(filename or "document").stem
    return _STORED_SUFFIX.sub("", stem) or "document"


def store(
    data: bytes | BinaryIO,
    filename: str,
    upload_dir: Path | None = None,
) -> tuple[Path, str, int]:
    """
    Write an upload to disk. Returns (path, sha256, size).

    The stored name carries the content hash, so re-uploading the same file
    overwrites its own copy instead of accumulating "report (3).docx".
    """
    if not isinstance(data, (bytes, bytearray)):
        data = data.read()

    size = len(data)
    if size == 0:
        raise UploadRejected(f"{filename} is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"{filename} is {size / 1e6:.0f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormat(
            f"{suffix or filename} is not supported. "
            f"Supported formats: {', '.join(SUPPORTED_SUFFIXES)}"
        )

    digest = hashlib.sha256(data).hexdigest()

    directory = Path(upload_dir) if upload_dir else UPLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)

    name = safe_name(Path(filename).stem)
    path = directory / f"{name}-{digest[:12]}{suffix}"
    path.write_bytes(data)

    return path, digest, size


def find_by_hash(digest: str, upload_dir: Path | None = None) -> Path | None:
    """
    The stored file for a content hash, or None.

    `store` names files "<safe-stem>-<first 12 of sha256><suffix>", so the hash
    is recoverable from the name. That is what lets the API stay stateless: the
    browser holds only a hash, and the backend finds the document from it
    without a session table.
    """
    if not digest:
        return None

    directory = Path(upload_dir) if upload_dir else UPLOAD_DIR
    if not directory.is_dir():
        return None

    for path in directory.glob(f"*-{digest[:12]}.*"):
        if path.is_file():
            return path
    return None
