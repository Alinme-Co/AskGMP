"""
pdf.py — PDF into records, at two levels of detail.

    STRUCTURAL   pymupdf4llm renders the document as markdown, one record per
                 page, with plain PyMuPDF as the safety net. Markdown is the
                 point: it preserves the heading structure a regulation is
                 cited by, and it is what the knowledge base's 1,844 indexed
                 chunks were built from. OCR runs only when a document yields no
                 text at all, because it is slow and needs Tesseract.

    RICH         app/documents/pdf_engine.py: read the selectable text layer,
                 decide from how little of it there is whether the page was
                 scanned, OCR the whole page when it was, and otherwise OCR only
                 the embedded pictures — where a flowchart or a signed form can
                 hold text that exists nowhere in the text layer. What an upload
                 gets, because an uploaded SOP is as likely to be a scan as not.

Keeping both is deliberate rather than a leftover. Running pdf_engine over the
corpus would lose the markdown headings and force a full re-extraction and
re-index of every document; running pymupdf4llm over an uploaded scan would
return nothing at all.

The rich path also splits a page's prose from its figures. pdf_engine returns
both on one PageContent; a diagram's OCR is a self-contained caption, and
merging it into the surrounding paragraph produces a chunk that reads as
nonsense at retrieval time.
"""

from __future__ import annotations

import logging

from app.documents.models import (
    KIND_IMAGE,
    KIND_PAGE,
    drop_empty,
    figure_locator,
    page_locator,
    record,
)
from app.documents.text_utils import normalise_text

logger = logging.getLogger(__name__)

SUFFIXES = (".pdf",)

# pymupdf4llm stalls for minutes on pages with thousands of vector drawings.
# Above this count it skips the page, and the plain-PyMuPDF fallback picks it up.
GRAPHICS_LIMIT = 100


def extract(path: str, use_ocr: bool = False, ocr=None,
            profile: str = "rich", password: str | None = None,
            **_) -> tuple[list[dict], str]:
    if profile == "structural":
        return _extract_structural(path, use_ocr=use_ocr)
    return _extract_rich(path, use_ocr=use_ocr, password=password), "extracted"


# ── Structural: pymupdf4llm markdown, PyMuPDF fallback ────────────────────────
def _extract_structural(path: str, use_ocr: bool = False) -> tuple[list[dict], str]:
    """
    The fallback covers two distinct failures: to_markdown raising outright, and
    the quieter per-page case where a page exceeds GRAPHICS_LIMIT and comes back
    empty. Both end with real text on the page.
    """
    import pymupdf

    pages: list[dict] = []
    status = "extracted"

    try:
        import pymupdf4llm

        blocks = pymupdf4llm.to_markdown(
            path,
            page_chunks    = True,
            graphics_limit = GRAPHICS_LIMIT,
            show_progress  = False,
        )
        for block in blocks:
            number = (block.get("metadata") or {}).get("page", len(pages) + 1)
            pages.append({"text": block.get("text", "") or "", "page": number})
    except Exception as exc:                     # noqa: BLE001 - want the fallback
        status = f"markdown-fallback: {type(exc).__name__}"
        pages = []

    fallback_pages = 0
    try:
        with pymupdf.open(path) as document:
            if not pages:
                pages = [
                    {"text": document[i].get_text() or "", "page": i + 1}
                    for i in range(document.page_count)
                ]
                fallback_pages = len(pages)
            else:
                for page in pages:
                    if page["text"].strip():
                        continue
                    index = (page["page"] or 0) - 1
                    if 0 <= index < document.page_count:
                        page["text"] = document[index].get_text() or ""
                        if page["text"].strip():
                            fallback_pages += 1

            # OCR only where there is genuinely nothing to read — it is slow and
            # needs Tesseract, so it never runs on a document that extracted.
            if use_ocr and not any(page["text"].strip() for page in pages):
                ocr_pages = []
                for index in range(document.page_count):
                    textpage = document[index].get_textpage_ocr(flags=0, full=True)
                    ocr_pages.append({
                        "text": document[index].get_text(textpage=textpage) or "",
                        "page": index + 1,
                    })
                if any(page["text"].strip() for page in ocr_pages):
                    pages = ocr_pages
                    status = "ocr"
    except Exception as exc:                     # noqa: BLE001
        if not pages:
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

    if fallback_pages and status == "extracted":
        status = f"extracted (+{fallback_pages}p fallback)"

    ocr_used = status == "ocr"
    records = [
        record(
            normalise_text(page["text"]),
            kind=KIND_PAGE,
            page=page["page"],
            atomic=False,          # consecutive pages are one passage
            ocr=ocr_used,
        )
        for page in pages
    ]

    return drop_empty(records), status


# ── Rich: text layer, scanned-page OCR, per-picture OCR ───────────────────────
def _extract_rich(path: str, use_ocr: bool = True,
                  password: str | None = None) -> list[dict]:
    from app.documents.pdf_engine import DocumentReader

    reader = DocumentReader(use_ocr=use_ocr)
    pages = reader.read_pdf_pages(path, password=password)

    records: list[dict] = []

    for page in pages:
        number = page.number
        locator = page_locator(number)

        # The page's own prose: the text layer, or the whole-page OCR when the
        # page turned out to be a scan.
        body = "\n\n".join(
            part for part in (page.text.strip(), page.ocr_text.strip()) if part
        )
        if body:
            records.append(record(
                body,
                kind=KIND_PAGE,
                page=number,
                locator=locator,
                atomic=False,          # consecutive pages are one passage
                ocr=bool(page.ocr_text.strip()),
                scanned=page.is_scanned or None,
            ))

        # Each figure separately, so a citation can point at the figure rather
        # than at the page it happened to sit on.
        for image in page.images:
            if not image.has_text:
                continue
            records.append(record(
                image.text,
                kind=KIND_IMAGE,
                page=number,
                locator=figure_locator(image.index, page=number),
                atomic=True,
                ocr=True,
                bbox=[round(value, 1) for value in image.bbox],
            ))

    return drop_empty(records)
