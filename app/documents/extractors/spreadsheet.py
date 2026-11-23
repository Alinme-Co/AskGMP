"""
spreadsheet.py — Excel workbooks into records, one row at a time.

The old excel.py walked cells to hand them to a translator, so a cell was the
unit and its neighbours were irrelevant. For retrieval a lone cell is close to
useless: "35 °C" answers nothing without the row that says what it measures and
the header that says what the column is. So a row is the unit here, rendered
with its column headers:

    Parameter: Storage temperature | Limit: 2-8 °C | Ref: SOP-114

That reads as a sentence, embeds meaningfully, and still carries the exact cell
reference of every value in `meta` so a citation can name one.

Two other things the translation reader had no reason to do:

* **Pictures are OCR-ed.** A workbook's evidence is often a pasted screenshot
  of an instrument reading. openpyxl exposes them through `ws._images`, with
  the anchor telling us which cell each one sits on.
* **Merged cells are resolved.** openpyxl reports the value on the top-left
  cell of a merge and None on the rest, so a merged header would otherwise
  vanish from every row under it.
"""

from __future__ import annotations

import logging
from typing import Any

from app.documents.ocr import ImageOCR
from app.documents.models import (
    KIND_IMAGE,
    KIND_ROW,
    KIND_SHEET,
    drop_empty,
    figure_locator,
    record,
    sheet_locator,
)

logger = logging.getLogger(__name__)

SUFFIXES = (".xlsx", ".xlsm", ".xltx", ".xltm")

# A header row is only useful if it is mostly text. A sheet that starts with
# numbers has no headers, and inventing them would mislabel every row.
MAX_HEADER_SCAN = 5
MAX_CELLS_PER_ROW = 60


def extract(path: str, use_ocr: bool = True, ocr: ImageOCR | None = None,
            **_) -> tuple[list[dict], str]:
    import openpyxl

    # data_only: formulas are not evidence; their last computed value is.
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=False)
    engine = ocr or ImageOCR(enabled=use_ocr)

    records: list[dict] = []
    try:
        for sheet in workbook.worksheets:
            if sheet.sheet_state != "visible":
                # Hidden sheets are usually scratch calculations, and their
                # content is not what the author published.
                continue
            records.extend(_read_sheet(sheet, engine))
    finally:
        workbook.close()

    return drop_empty(records), "extracted"


def _read_sheet(sheet, engine: ImageOCR) -> list[dict]:
    name = sheet.title
    merged = _merged_lookup(sheet)

    rows = list(sheet.iter_rows())
    headers = _find_headers(rows, merged)

    records: list[dict] = []

    if headers:
        records.append(record(
            "Columns: " + " | ".join(value for value in headers.values() if value),
            kind=KIND_SHEET,
            section=sheet_locator(name),
            locator=sheet_locator(name),
            atomic=True,
            sheet=name,
        ))

    header_row_index = _header_row_index(rows, headers, merged)

    for row in rows:
        if not row:
            continue
        number = row[0].row
        if header_row_index is not None and number == header_row_index:
            continue

        rendered, cells = _render_row(row, headers, merged)
        if not rendered:
            continue

        records.append(record(
            rendered,
            kind=KIND_ROW,
            section=sheet_locator(name, row=number),
            locator=sheet_locator(name, row=number),
            # A spreadsheet row is a complete statement; gluing two unrelated
            # rows together makes a chunk that belongs to neither.
            atomic=True,
            sheet=name,
            row=number,
            cells=cells[:MAX_CELLS_PER_ROW],
        ))

    records.extend(_read_images(sheet, engine))
    return records


def _merged_lookup(sheet) -> dict[tuple[int, int], Any]:
    """Every covered cell mapped to the value of its merge's top-left cell."""
    lookup: dict[tuple[int, int], Any] = {}
    for cell_range in getattr(sheet, "merged_cells", []).ranges if hasattr(sheet, "merged_cells") else []:
        try:
            top_left = sheet.cell(row=cell_range.min_row, column=cell_range.min_col).value
        except Exception:                         # noqa: BLE001
            continue
        if top_left is None:
            continue
        for row in range(cell_range.min_row, cell_range.max_row + 1):
            for column in range(cell_range.min_col, cell_range.max_col + 1):
                lookup[(row, column)] = top_left
    return lookup


def _value(cell, merged: dict[tuple[int, int], Any]) -> str:
    value = cell.value
    if value is None:
        value = merged.get((cell.row, cell.column))
    if value is None:
        return ""
    return str(value).strip()


def _find_headers(rows, merged) -> dict[int, str]:
    """
    Column index -> header label, from the first mostly-textual row.

    Only the first few rows are considered: a header that appears halfway down
    a sheet is a section break, not a header, and treating it as one would
    label every row below it wrongly.
    """
    for row in rows[:MAX_HEADER_SCAN]:
        values = {cell.column: _value(cell, merged) for cell in row}
        filled = {index: text for index, text in values.items() if text}
        if len(filled) < 2:
            continue
        textual = sum(1 for text in filled.values() if not _is_number(text))
        if textual >= max(2, len(filled) * 0.6):
            return filled
    return {}


def _header_row_index(rows, headers: dict[int, str], merged) -> int | None:
    if not headers:
        return None
    for row in rows[:MAX_HEADER_SCAN]:
        values = {cell.column: _value(cell, merged) for cell in row}
        if {k: v for k, v in values.items() if v} == headers:
            return row[0].row if row else None
    return None


def _is_number(text: str) -> bool:
    try:
        float(text.replace(",", "").replace("%", ""))
        return True
    except ValueError:
        return False


def _render_row(row, headers: dict[int, str], merged) -> tuple[str, list[dict]]:
    """One row as "Header: value | Header: value", plus its cell references."""
    parts: list[str] = []
    cells: list[dict] = []

    for cell in row:
        text = _value(cell, merged)
        if not text:
            continue

        header = headers.get(cell.column, "")
        parts.append(f"{header}: {text}" if header else text)
        cells.append({
            "ref": cell.coordinate,
            "column": header or cell.column_letter,
            "value": text[:200],
        })

    return " | ".join(parts), cells


def _read_images(sheet, engine: ImageOCR) -> list[dict]:
    """
    OCR the pictures anchored in a sheet.

    Skipped entirely when OCR is off, so a workbook full of screenshots does
    not silently cost a minute of model time on a run that asked for text only.
    """
    if not engine.available:
        return []

    records: list[dict] = []
    images = getattr(sheet, "_images", []) or []

    for index, image in enumerate(images, start=1):
        try:
            data = image._data()               # noqa: SLF001 - openpyxl's only accessor
        except Exception as exc:               # noqa: BLE001
            logger.debug("Could not read an image from sheet %s: %s", sheet.title, exc)
            continue

        text = engine.text_from_bytes(data)
        if not text:
            continue

        anchor = _anchor_cell(image)
        records.append(record(
            text,
            kind=KIND_IMAGE,
            section=sheet_locator(sheet.title, cell=anchor),
            locator=figure_locator(index, sheet=sheet.title),
            atomic=True,
            ocr=True,
            sheet=sheet.title,
            cell=anchor,
        ))

    return records


def _anchor_cell(image) -> str | None:
    """The cell reference a picture is pinned to, when openpyxl exposes one."""
    anchor = getattr(image, "anchor", None)
    marker = getattr(anchor, "_from", None)
    if marker is None:
        return None
    try:
        from openpyxl.utils import get_column_letter

        return f"{get_column_letter(marker.col + 1)}{marker.row + 1}"
    except Exception:                             # noqa: BLE001
        return None
