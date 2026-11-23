"""
word.py — .docx into records: prose, tables, headers/footers and OCR-ed figures.

The old word_html.py walked paragraphs and tables to rebuild them as styled
HTML for a translator, so formatting was the product. Here formatting matters
only where it carries meaning — a Heading 2 tells you the section a paragraph
belongs to, and that section is what a citation should name — so styles are
read for structure and then discarded.

Body order matters and python-docx does not give it away. `document.paragraphs`
and `document.tables` are two separate lists, so reading them in turn puts every
table at the end and detaches it from the paragraph that introduced it.
`_iter_body` walks the underlying XML instead, which keeps a table next to the
sentence that says what it contains.
"""

from __future__ import annotations

import logging

from app.documents.ocr import ImageOCR
from app.documents.models import (
    KIND_HEADER,
    KIND_HEADING,
    KIND_IMAGE,
    KIND_PARAGRAPH,
    KIND_TABLE,
    drop_empty,
    figure_locator,
    record,
    table_locator,
)

logger = logging.getLogger(__name__)

SUFFIXES = (".docx", ".docm")


def extract(path: str, use_ocr: bool = True, ocr: ImageOCR | None = None,
            **_) -> tuple[list[dict], str]:
    import docx

    document = docx.Document(path)
    engine = ocr or ImageOCR(enabled=use_ocr)

    records: list[dict] = []
    section: str | None = None
    table_index = 0

    for kind, item in _iter_body(document):
        if kind == "paragraph":
            text = (item.text or "").strip()
            if not text:
                continue

            heading_level = _heading_level(item)
            if heading_level:
                section = text
                records.append(record(
                    text,
                    kind=KIND_HEADING,
                    section=section,
                    locator=text,
                    atomic=False,
                    level=heading_level,
                ))
            else:
                records.append(record(
                    text,
                    kind=KIND_PARAGRAPH,
                    section=section,
                    locator=section,
                    atomic=False,      # flowing prose; merge freely
                    style=_style_name(item),
                ))

        elif kind == "table":
            table_index += 1
            rendered = _render_table(item)
            if not rendered:
                continue
            locator = table_locator(table_index)
            records.append(record(
                rendered,
                kind=KIND_TABLE,
                section=section or locator,
                locator=locator,
                # A table is one object. Splitting it across chunks, or gluing
                # a paragraph onto it, loses the header row that gives every
                # cell its meaning.
                atomic=True,
                table=table_index,
                rows=len(item.rows),
                columns=len(item.columns),
            ))

    records.extend(_read_headers_footers(document))
    records.extend(_read_images(document, engine))

    return drop_empty(records), "extracted"


# ── Body order ────────────────────────────────────────────────────────────────
def _iter_body(document):
    """Paragraphs and tables in the order they appear in the document."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            yield "paragraph", Paragraph(child, document)
        elif tag == "tbl":
            yield "table", Table(child, document)


def _style_name(paragraph) -> str | None:
    try:
        return paragraph.style.name
    except Exception:                             # noqa: BLE001
        return None


def _heading_level(paragraph) -> int | None:
    """The outline level of a heading paragraph, or None for body text."""
    name = (_style_name(paragraph) or "").strip()
    if name.lower().startswith("heading"):
        tail = name.split()[-1]
        return int(tail) if tail.isdigit() else 1
    if name in {"Title", "Subtitle"}:
        return 1
    return None


# ── Tables ────────────────────────────────────────────────────────────────────
def _render_table(table) -> str:
    """
    A table as pipe-separated rows, header first.

    Kept as text rather than turned into prose: a GMP specification table is
    read as a table, and the column alignment is part of what it says.
    """
    lines: list[str] = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            text = " ".join(
                paragraph.text.strip()
                for paragraph in cell.paragraphs
                if paragraph.text.strip()
            )
            cells.append(text)
        # A row of empty cells is a spacer, not data.
        if any(cell for cell in cells):
            lines.append(" | ".join(cells))

    # Deduplicate the merged-cell repetition python-docx produces: a cell
    # spanning three columns is reported three times on the same row.
    return "\n".join(_collapse_repeats(line) for line in lines)


def _collapse_repeats(line: str) -> str:
    parts = [part.strip() for part in line.split("|")]
    out: list[str] = []
    for part in parts:
        if not out or part != out[-1]:
            out.append(part)
    return " | ".join(out)


# ── Headers, footers, images ──────────────────────────────────────────────────
def _read_headers_footers(document) -> list[dict]:
    """
    Running headers and footers.

    Worth keeping in a regulated document: the footer is usually where the SOP
    number, revision and effective date live, and that is exactly the metadata
    someone is trying to find.
    """
    records: list[dict] = []
    seen: set[str] = set()

    for index, section in enumerate(document.sections, start=1):
        for part, kind in ((section.header, "header"), (section.footer, "footer")):
            if part is None:
                continue
            text = "\n".join(
                paragraph.text.strip()
                for paragraph in part.paragraphs
                if paragraph.text.strip()
            ).strip()
            # Repeated on every section; once is enough.
            if not text or text in seen:
                continue
            seen.add(text)
            records.append(record(
                text,
                kind=KIND_HEADER,
                section=kind.capitalize(),
                locator=f"{kind.capitalize()} (section {index})",
                atomic=True,
                part=kind,
            ))

    return records


def _read_images(document, engine: ImageOCR) -> list[dict]:
    """OCR every embedded picture, in document order."""
    if not engine.available:
        return []

    records: list[dict] = []
    index = 0

    for rel_id, part in document.part.related_parts.items():
        if "image" not in getattr(part, "content_type", ""):
            continue
        index += 1
        try:
            data = part.blob
        except Exception as exc:                  # noqa: BLE001
            logger.debug("Could not read image %s: %s", rel_id, exc)
            continue

        text = engine.text_from_bytes(data)
        if not text:
            continue

        records.append(record(
            text,
            kind=KIND_IMAGE,
            locator=figure_locator(index),
            atomic=True,
            ocr=True,
            figure=index,
        ))

    return records
