"""
exceptions.py — the errors AskGMP raises on purpose.

Defined here rather than in the modules that raise them because they cross
package boundaries: a reader raises DocumentReadFailed, the upload service
catches it, and the API turns it into a message. Putting them in one place
keeps `app.uploads` from having to import `app.documents` just for an
exception type.
"""

from __future__ import annotations


class AskGMPError(Exception):
    """Base class, so a caller can catch everything this project raises."""


class UnsupportedFormat(AskGMPError):
    """Raised for a file extension no extractor claims."""


class DocumentReadFailed(AskGMPError):
    """Raised when an extractor could not read a file it should have understood."""


class UploadRejected(AskGMPError):
    """The file cannot be accepted — empty, or too large."""
