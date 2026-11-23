"""
app.documents.extractors — one extraction subsystem, two callers.

    from app.documents.extractors import read_document, extract_records

    read_document("spec.xlsx")            # uploads: records only
    extract_records("annex15.pdf")        # knowledge base: (records, status)

Both go through the same registry, the same dispatch and the same record model
(app/documents/models.py). There used to be two stacks — `ingestion/extractors`
for the corpus and `documents/readers` for uploads — which meant two answers to
"is this JSON a Q&A collection?" and two HTML cleaners that could drift apart.

Profiles
--------
The two callers want genuinely different things from the same file, so the
dispatcher takes a profile rather than pretending they are identical:

    STRUCTURAL   what the knowledge base indexes. Fast, layout-aware, no OCR
                 unless a page has no text layer at all. pymupdf4llm renders a
                 PDF as markdown, which is what preserves the heading structure
                 a regulation is cited by.

    RICH         what an upload gets. Slower and thorough: scanned pages are
                 detected and OCR-ed, pictures inside a .docx or .pptx are read,
                 tables and rows become their own records with cell-level
                 provenance.

Most formats ignore the profile entirely — HTML, Markdown, JSON, CSV and plain
text extract the same way for both, and that shared code is the bulk of what
this package removed. PDF and PowerPoint are the two where the difference is
real, and each keeps both strategies in one module rather than in two packages.

Extractor modules are imported lazily. pymupdf, openpyxl, python-docx,
python-pptx and RapidOCR are heavy and mutually irrelevant — a run that reads
one CSV should not pay to import an ONNX runtime.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from app.core.exceptions import DocumentReadFailed, UnsupportedFormat
from app.documents.models import summarise
from app.documents.ocr import ImageOCR, shared_ocr

logger = logging.getLogger(__name__)

#: What the knowledge-base indexer asks for: structure over thoroughness.
PROFILE_STRUCTURAL = "structural"

#: What an uploaded document asks for: thoroughness over speed.
PROFILE_RICH = "rich"


# suffix -> (module path, human label). One row per supported format; adding a
# format means adding an extractor module and a line here.
_REGISTRY: dict[str, tuple[str, str]] = {}


def _register(module: str, label: str, suffixes: tuple[str, ...]) -> None:
    for suffix in suffixes:
        _REGISTRY[suffix] = (f"app.documents.extractors.{module}", label)


_register("pdf", "PDF", (".pdf",))
_register("word", "Word", (".docx", ".docm"))
_register("powerpoint", "PowerPoint", (".pptx", ".pptm"))
_register("spreadsheet", "Excel", (".xlsx", ".xlsm", ".xltx", ".xltm"))
_register("html", "HTML", (".html", ".htm", ".xhtml"))
_register("xliff", "XLIFF", (".xliff", ".xlf", ".sdlxliff", ".mxliff"))
_register("text", "Markdown", (".md", ".markdown"))
_register("json", "JSON", (".json",))
_register("text", "CSV", (".csv", ".tsv"))
_register("text", "Text", (".txt", ".log", ".rst"))


SUPPORTED_SUFFIXES = tuple(sorted(_REGISTRY))

#: Extensions without the dot, for st.file_uploader's `type=` argument.
UPLOAD_TYPES = [suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES]

#: Formats that can carry text inside pictures, and so benefit from OCR.
OCR_CAPABLE = {".pdf", ".docx", ".docm", ".pptx", ".pptm", ".xlsx", ".xlsm"}


def format_label(suffix: str) -> str:
    entry = _REGISTRY.get(suffix.lower())
    return entry[1] if entry else suffix.lstrip(".").upper()


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _REGISTRY


def _module_for(suffix: str):
    entry = _REGISTRY.get(suffix)
    if entry is None:
        raise UnsupportedFormat(
            f"{suffix or 'this file'} is not a supported format. "
            f"Supported: {', '.join(SUPPORTED_SUFFIXES)}"
        )
    module_path, label = entry
    return importlib.import_module(module_path), label


# ── The knowledge-base entry point ────────────────────────────────────────────
def extract_records(
    path: str | Path,
    suffix: str | None = None,
    use_ocr: bool = False,
    profile: str = PROFILE_STRUCTURAL,
) -> tuple[list[dict], str]:
    """
    Records plus a status string describing how extraction went.

    The status is what the indexer prints and what tells a fallback path apart
    from a clean read — "extracted", "extracted (q&a)", "ocr",
    "markdown-fallback: RuntimeError".
    """
    path = Path(path)
    suffix = (suffix or path.suffix).lower()
    module, _label = _module_for(suffix)

    return module.extract(
        str(path), suffix=suffix, use_ocr=use_ocr, ocr=None, profile=profile
    )


# ── The uploads entry point ───────────────────────────────────────────────────
def read_document(
    path: str | Path,
    use_ocr: bool = True,
    ocr: ImageOCR | None = None,
    profile: str = PROFILE_RICH,
    **kwargs,
) -> list[dict]:
    """
    Read any supported document into records.

    Raises UnsupportedFormat for an unknown extension and DocumentReadFailed
    when an extractor cannot cope with the file — a corrupt upload should
    surface as a clear message, not as an empty result that looks like a blank
    document.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    module, label = _module_for(suffix)

    if not path.is_file():
        raise DocumentReadFailed(f"File not found: {path}")

    # OCR is only built for formats that can hold pictures, and only when asked.
    # The shared engine rather than a fresh one: loading the three ONNX models
    # costs a second or two, and a batch of five uploads would otherwise pay
    # that five times over.
    engine = ocr
    if engine is None and use_ocr and suffix in OCR_CAPABLE:
        engine = shared_ocr(enabled=True)

    try:
        records, _status = module.extract(
            str(path), suffix=suffix, use_ocr=use_ocr, ocr=engine,
            profile=profile, **kwargs,
        )
    except Exception as exc:                      # noqa: BLE001
        raise DocumentReadFailed(
            f"Could not read this {label} file: {type(exc).__name__}: {exc}"
        ) from exc

    return records


def read_summary(path: str | Path, **kwargs) -> tuple[list[dict], dict]:
    """Records plus the counts a UI wants to show beside them."""
    records = read_document(path, **kwargs)
    return records, summarise(records)


__all__ = [
    "PROFILE_STRUCTURAL",
    "PROFILE_RICH",
    "SUPPORTED_SUFFIXES",
    "UPLOAD_TYPES",
    "OCR_CAPABLE",
    "DocumentReadFailed",
    "UnsupportedFormat",
    "extract_records",
    "read_document",
    "read_summary",
    "format_label",
    "is_supported",
]
