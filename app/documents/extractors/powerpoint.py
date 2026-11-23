"""
powerpoint.py — .pptx into records, at two levels of detail.

Both strategies agree on the central point: **a slide is one argument**, so a
slide is one atomic record and its bullets are never split apart or merged with
the next slide's.

    STRUCTURAL   title, body text and speaker notes as a single record per
                 slide. What the knowledge base indexes — MHRA's OOS guidance is
                 published only as a deck — and what the 1,844 points in Qdrant
                 were built from.

    RICH         the same slide record, plus each table and each picture as
                 records of their own. An uploaded deck's process flow is
                 frequently a diagram with the steps drawn inside it, and OCR is
                 the only way to read it.

The recursive walk over grouped shapes is the part everyone gets wrong first: a
shape whose type is GROUP has no text frame of its own, so a reader that only
checks `has_text_frame` silently drops every label inside a grouped diagram —
which in a GMP deck is usually the whole process flow.
"""

from __future__ import annotations

import logging

from app.documents.models import (
    KIND_IMAGE,
    KIND_NOTES,
    KIND_SLIDE,
    KIND_TABLE,
    drop_empty,
    figure_locator,
    record,
    slide_locator,
    table_locator,
)
from app.documents.ocr import ImageOCR
from app.documents.text_utils import _WS_RE, normalise_text

logger = logging.getLogger(__name__)

SUFFIXES = (".pptx", ".pptm")


def extract(path: str, use_ocr: bool = True, ocr: ImageOCR | None = None,
            profile: str = "rich", **_) -> tuple[list[dict], str]:
    if profile == "structural":
        return _extract_structural(path), "extracted"
    return _extract_rich(path, use_ocr=use_ocr, ocr=ocr), "extracted"


# ── Structural: one record per slide ──────────────────────────────────────────
def _extract_structural(path: str) -> list[dict]:
    """
    Title, body and notes in one record per slide.

    Speaker notes are folded into the slide rather than kept separate because a
    guidance deck often carries the actual explanation in the notes and only a
    heading on the slide; separating them would index the heading alone.
    """
    from pptx import Presentation

    presentation = Presentation(path)
    records: list[dict] = []

    for number, slide in enumerate(presentation.slides, 1):
        title = ""
        try:
            if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
                title = _WS_RE.sub(" ", slide.shapes.title.text_frame.text).strip()
        except (AttributeError, ValueError):
            title = ""

        lines: list[str] = []
        for shape in _iter_shapes(slide.shapes):
            if not getattr(shape, "has_text_frame", False):
                continue
            body = shape.text_frame.text.strip()
            if not body or body == title:
                continue
            lines.append(body)

        notes = _notes(slide)
        if notes:
            lines.append(f"Speaker notes: {notes}")

        section = f"Slide {number}" + (f" — {title}" if title else "")
        parts = ([title] if title else []) + lines
        text = normalise_text("\n\n".join(parts))
        if text:
            records.append(record(
                text,
                kind=KIND_SLIDE,
                page=number,             # the slide number plays the part of a page
                section=section,
                atomic=True,
            ))

    return records


def _iter_shapes(shapes):
    """Flatten grouped shapes, which python-pptx nests."""
    for shape in shapes:
        if getattr(shape, "shape_type", None) is not None and hasattr(shape, "shapes"):
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


# ── Rich: slide, notes, tables and OCR-ed pictures ────────────────────────────
def _extract_rich(path: str, use_ocr: bool = True,
                  ocr: ImageOCR | None = None) -> list[dict]:
    from pptx import Presentation

    presentation = Presentation(path)
    engine = ocr or ImageOCR(enabled=use_ocr)

    records: list[dict] = []

    for number, slide in enumerate(presentation.slides, start=1):
        title = _title(slide)
        locator = slide_locator(number, title)

        body: list[str] = []
        tables: list = []
        pictures: list = []
        _walk(slide.shapes, body, tables, pictures, title)

        text = "\n\n".join(part for part in ([title] if title else []) + body if part)
        if text.strip():
            records.append(record(
                text,
                kind=KIND_SLIDE,
                page=number,               # a slide numbers like a page
                section=locator,
                locator=locator,
                atomic=True,               # one slide, one argument
                slide=number,
                title=title,
            ))

        notes = _notes(slide)
        if notes:
            records.append(record(
                notes,
                kind=KIND_NOTES,
                page=number,
                section=f"{locator} (speaker notes)",
                locator=f"{locator} (speaker notes)",
                atomic=True,
                slide=number,
            ))

        for index, table in enumerate(tables, start=1):
            rendered = _render_table(table)
            if not rendered:
                continue
            table_label = table_locator(index, slide=number)
            records.append(record(
                rendered,
                kind=KIND_TABLE,
                page=number,
                section=table_label,
                locator=table_label,
                atomic=True,
                slide=number,
                table=index,
            ))

        if engine.available:
            for index, picture in enumerate(pictures, start=1):
                text = _ocr_picture(picture, engine)
                if not text:
                    continue
                figure = figure_locator(
                    index, slide=number, shape=_shape_name(picture)
                )
                records.append(record(
                    text,
                    kind=KIND_IMAGE,
                    page=number,
                    section=figure,
                    locator=figure,
                    atomic=True,
                    ocr=True,
                    slide=number,
                    figure=index,
                ))

    return drop_empty(records)


# ── Shape traversal ───────────────────────────────────────────────────────────
def _walk(shapes, body: list[str], tables: list, pictures: list, title: str | None) -> None:
    """
    Collect text, tables and pictures, descending into grouped shapes.

    A group has no text frame of its own; its children do. Checking only
    `has_text_frame` at the top level loses every label in a grouped diagram.
    """
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    for shape in shapes:
        try:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                _walk(shape.shapes, body, tables, pictures, title)
                continue
        except (AttributeError, ValueError):
            pass                                  # odd shape; fall through

        if getattr(shape, "has_table", False):
            tables.append(shape.table)
            continue

        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            pictures.append(shape)
            continue

        if getattr(shape, "has_text_frame", False):
            text = (shape.text_frame.text or "").strip()
            # The title is emitted separately; repeating it reads badly.
            if text and text != title:
                body.append(text)


def _title(slide) -> str | None:
    try:
        placeholder = slide.shapes.title
        if placeholder is not None and placeholder.has_text_frame:
            return (placeholder.text_frame.text or "").strip() or None
    except (AttributeError, ValueError):
        pass
    return None


def _notes(slide) -> str:
    try:
        if not slide.has_notes_slide:
            return ""
        return (slide.notes_slide.notes_text_frame.text or "").strip()
    except (AttributeError, ValueError):
        return ""


def _shape_name(shape) -> str | None:
    name = (getattr(shape, "name", "") or "").strip()
    # python-pptx names every picture "Picture 3" by default; that adds nothing.
    return name if name and not name.lower().startswith("picture") else None


def _render_table(table) -> str:
    lines = []
    for row in table.rows:
        cells = [(cell.text or "").strip() for cell in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _ocr_picture(picture, engine: ImageOCR) -> str:
    try:
        blob = picture.image.blob
    except Exception as exc:                      # noqa: BLE001
        # Charts and linked pictures have no blob; nothing to OCR.
        logger.debug("Picture has no readable image data: %s", exc)
        return ""
    return engine.text_from_bytes(blob)
