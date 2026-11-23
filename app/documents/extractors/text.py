"""
text.py — Markdown, plain text, CSV and TSV.

**Markdown** becomes one record per heading section, headings kept verbatim.
Markdown structure is signal, not noise: the downloader wrote these files
precisely because the source had a section or question structure worth
preserving, so the heading stays in the text where the embedding can see it and
in `section` where the citation can use it.

**CSV** gets the same treatment as a spreadsheet row: each line rendered with
its column headers, so a value is never indexed without the column that says
what it is.

**Plain text** has no structure to find, so it becomes one record and the
chunker decides where the boundaries go.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from app.documents.models import KIND_ROW, KIND_TEXT, drop_empty, record
from app.documents.text_utils import normalise_text

TEXT_SUFFIXES     = (".txt", ".log", ".rst")
MARKDOWN_SUFFIXES = (".md", ".markdown")
CSV_SUFFIXES      = (".csv", ".tsv")
SUFFIXES = TEXT_SUFFIXES + MARKDOWN_SUFFIXES + CSV_SUFFIXES

MAX_CSV_FIELD = 200

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


def extract(path: str, suffix: str | None = None, **_) -> tuple[list[dict], str]:
    suffix = (suffix or Path(path).suffix).lower()

    if suffix in CSV_SUFFIXES:
        return _read_csv(path, delimiter="\t" if suffix == ".tsv" else ","), "extracted"
    if suffix in MARKDOWN_SUFFIXES:
        return _read_markdown(path), "extracted"
    return _read_text(path), "extracted"


# ── Markdown ──────────────────────────────────────────────────────────────────
def _read_markdown(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    records: list[dict] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            records.append(record(
                normalise_text(body),
                kind=KIND_TEXT,
                section=section,
                locator=section,
            ))
        buffer.clear()

    for line in text.splitlines():
        match = _MD_HEADING_RE.match(line.strip())
        if match:
            flush()
            section = match.group(2).strip()[:160]
            buffer.append(line.rstrip())
        else:
            buffer.append(line.rstrip())

    flush()

    if not records:
        cleaned = normalise_text(text)
        records = [record(cleaned, kind=KIND_TEXT)] if cleaned else []

    return drop_empty(records)


# ── Plain text ────────────────────────────────────────────────────────────────
def _read_text(path: str) -> list[dict]:
    text = normalise_text(Path(path).read_text(encoding="utf-8", errors="replace"))
    return [record(text, kind=KIND_TEXT)] if text else []


# ── CSV / TSV ─────────────────────────────────────────────────────────────────
def _read_csv(path: str, delimiter: str = ",") -> list[dict]:
    """One record per row, each value labelled with its column header."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=delimiter + ",;\t|")
    except csv.Error:
        dialect = None                            # fall back to the delimiter given

    reader = (
        csv.reader(text.splitlines(), dialect)
        if dialect else
        csv.reader(text.splitlines(), delimiter=delimiter)
    )

    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []

    headers = [cell.strip() for cell in rows[0]]
    # A header row is only a header if it is text; a file that opens with
    # numbers is data from line one.
    has_headers = sum(1 for cell in headers if cell and not _is_number(cell)) >= max(
        1, len(headers) * 0.6
    )
    body = rows[1:] if has_headers else rows

    records: list[dict] = []
    for number, row in enumerate(body, start=2 if has_headers else 1):
        parts = []
        for index, cell in enumerate(row):
            value = cell.strip()[:MAX_CSV_FIELD]
            if not value:
                continue
            header = headers[index] if has_headers and index < len(headers) else ""
            parts.append(f"{header}: {value}" if header else value)

        if not parts:
            continue

        records.append(record(
            " | ".join(parts),
            kind=KIND_ROW,
            section=f"Row {number}",
            locator=f"Row {number}",
            atomic=True,
            row=number,
        ))

    return drop_empty(records)


def _is_number(text: str) -> bool:
    try:
        float(text.replace(",", "").replace("%", ""))
        return True
    except ValueError:
        return False
