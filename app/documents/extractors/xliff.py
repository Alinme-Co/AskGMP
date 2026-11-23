"""
xliff.py — translation interchange files.

An XLIFF is a list of `trans-unit` elements, each with a `source` and possibly a
`target`. Each unit becomes one atomic record — a segment is a complete sentence
by construction, which is exactly the unit retrieval wants — and the source text
is what gets indexed, with any existing target kept in `meta` rather than
concatenated into the text, so two languages never end up embedded as one
string.
"""

from __future__ import annotations

from pathlib import Path

from app.documents.models import KIND_SEGMENT, drop_empty, record

SUFFIXES = (".xliff", ".xlf", ".sdlxliff", ".mxliff")


def extract(path: str, **_) -> tuple[list[dict], str]:
    from bs4 import BeautifulSoup

    markup = Path(path).read_text(encoding="utf-8", errors="replace")
    # lxml-xml keeps the namespaced tags XLIFF uses; html.parser mangles them.
    try:
        soup = BeautifulSoup(markup, "lxml-xml")
    except Exception:                             # noqa: BLE001 - lxml missing
        soup = BeautifulSoup(markup, "html.parser")

    units = soup.find_all(["trans-unit", "unit", "segment"])
    file_node = soup.find("file")
    source_language = _attr(file_node, "source-language", "srcLang")
    target_language = _attr(file_node, "target-language", "trgLang")

    records: list[dict] = []

    for index, unit in enumerate(units, start=1):
        source = unit.find("source")
        if source is None:
            continue

        text = _plain(source)
        if not text:
            continue

        target = unit.find("target")
        identifier = unit.get("id") or str(index)
        locator = f"Segment {identifier}"

        records.append(record(
            text,
            kind=KIND_SEGMENT,
            section=locator,
            locator=locator,
            # A translation segment is one sentence by construction.
            atomic=True,
            segment=identifier,
            source_language=source_language,
            target_language=target_language,
            # Kept beside the text, never inside it: concatenating a
            # translation onto its source embeds two languages as one string.
            target_text=_plain(target) if target is not None else None,
        ))

    return drop_empty(records), "extracted"


def _plain(node) -> str:
    """Segment text with inline placeholder tags removed."""
    if node is None:
        return ""
    text = node.get_text(" ", strip=True)
    return " ".join(text.split())


def _attr(node, *names: str) -> str | None:
    if node is None:
        return None
    for name in names:
        value = node.get(name)
        if value:
            return str(value)
    return None
