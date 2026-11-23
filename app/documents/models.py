"""
models.py — the one record shape every extractor produces, and how provenance
is expressed.

    {"text": str, "page": int|None, "section": str|None,
     "atomic": bool, "meta": {...}}

There used to be two of these — one for knowledge-base ingestion, one for
uploads — with identical fields and separate constructors. They are the same
model, so there is now one, and both workflows build through `record()`. That
is what lets a chunk from an uploaded SOP and a chunk from a EudraLex annex
flow through the same chunker, the same context builder and the same citation
validator with no translation step.

`atomic` is the load-bearing field. A PDF page is a slice of continuous prose,
so consecutive pages may be merged to fill the token budget. A Q&A pair, a CFR
section, a slide or a spreadsheet row is not: merging "§ 211.84 Testing of
components" onto the end of "§ 211.82 Receipt and storage" produces a chunk
that answers neither question well and cites the wrong section.

`meta` carries structured provenance. The knowledge-base extractors only had to
say which page a passage came from, because their corpus is paginated
documents. An uploaded workbook has no pages — it has sheets, rows and cells —
and a slide deck has shapes and tables. `locator` is the short string a
citation shows, and every extractor builds it through the helpers below rather
than formatting its own, so the same kind of origin always reads the same way.
"""

from __future__ import annotations

from typing import Any

from app.documents.text_utils import normalise_text

# What produced a record. Kept small and closed: a retriever can filter on it,
# and a UI can group by it.
KIND_PAGE      = "page"        # a PDF page
KIND_PARAGRAPH = "paragraph"   # body prose
KIND_HEADING   = "heading"
KIND_TABLE     = "table"
KIND_CELL      = "cell"        # one spreadsheet cell worth its own record
KIND_ROW       = "row"         # a spreadsheet row rendered as one line
KIND_SHEET     = "sheet"
KIND_SLIDE     = "slide"
KIND_NOTES     = "notes"       # speaker notes
KIND_IMAGE     = "image"       # OCR of a picture, chart or diagram
KIND_HEADER    = "header"      # page header / footer
KIND_SEGMENT   = "segment"     # one XLIFF translation unit
KIND_TEXT      = "text"        # plain text / markdown section
KIND_QA        = "qa"          # one question-and-answer pair


#: Kept as an alias because it reads better inside the extractors that only
#: normalise whitespace, and because `record()` applies it on every text.
normalise = normalise_text


def record(
    text: str,
    *,
    kind: str = KIND_TEXT,
    page: int | None = None,
    section: str | None = None,
    atomic: bool = False,
    locator: str | None = None,
    ocr: bool = False,
    **provenance: Any,
) -> dict:
    """
    One extracted unit.

    Provenance keyword arguments land in `meta`, with None values dropped, so a
    caller can pass `sheet=..., row=...` unconditionally and only the ones that
    apply are recorded.
    """
    meta: dict[str, Any] = {"kind": kind}
    if locator:
        meta["locator"] = locator
    if ocr:
        meta["ocr"] = True
    meta.update({key: value for key, value in provenance.items() if value is not None})

    return {
        "text": normalise_text(text),
        "page": page,
        "section": section or locator,
        "atomic": atomic,
        "meta": meta,
    }


def drop_empty(records: list[dict]) -> list[dict]:
    return [item for item in records if item.get("text", "").strip()]


# ── Locators ──────────────────────────────────────────────────────────────────
# One function per origin, so "Slide 4" is never written "slide 4" somewhere
# else. These strings end up in front of a user as the answer's citation.
def sheet_locator(sheet: str, cell: str | None = None, row: int | None = None) -> str:
    if cell:
        return f"Sheet '{sheet}'!{cell}"
    if row is not None:
        return f"Sheet '{sheet}', row {row}"
    return f"Sheet '{sheet}'"


def page_locator(page: int) -> str:
    return f"Page {page}"


def slide_locator(number: int, title: str | None = None) -> str:
    return f"Slide {number}" + (f" — {title}" if title else "")


def table_locator(
    index: int,
    *,
    sheet: str | None = None,
    slide: int | None = None,
    page: int | None = None,
    row: int | None = None,
) -> str:
    where = ""
    if sheet:
        where = f" (sheet '{sheet}')"
    elif slide is not None:
        where = f" (slide {slide})"
    elif page is not None:
        where = f" (page {page})"
    label = f"Table {index}{where}"
    return f"{label}, row {row}" if row is not None else label


def figure_locator(
    index: int,
    *,
    page: int | None = None,
    slide: int | None = None,
    sheet: str | None = None,
    shape: str | None = None,
) -> str:
    """A picture, chart or diagram whose text came from OCR."""
    where = ""
    if page is not None:
        where = f" on page {page}"
    elif slide is not None:
        where = f" on slide {slide}"
    elif sheet:
        where = f" in sheet '{sheet}'"
    name = f" '{shape}'" if shape else ""
    return f"Figure {index}{name}{where}"


# ── Summaries ─────────────────────────────────────────────────────────────────
def summarise(records: list[dict]) -> dict[str, Any]:
    """Counts a UI can show without re-walking the records itself."""
    kinds: dict[str, int] = {}
    for item in records:
        kind = (item.get("meta") or {}).get("kind", KIND_TEXT)
        kinds[kind] = kinds.get(kind, 0) + 1

    pages = [item["page"] for item in records if item.get("page") is not None]

    return {
        "records": len(records),
        "characters": sum(len(item.get("text", "")) for item in records),
        "kinds": dict(sorted(kinds.items(), key=lambda pair: -pair[1])),
        "ocr_records": sum(
            1 for item in records if (item.get("meta") or {}).get("ocr")
        ),
        "pages": (min(pages), max(pages)) if pages else None,
    }
