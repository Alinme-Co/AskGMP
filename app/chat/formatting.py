"""
formatting.py — Turning payload values into the strings a citation shows.

Small, pure, and heavily tested, because every one of these functions sits
between a stored fact and something a user will read as a regulatory reference.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# The indexer writes section labels with the page appended —
# "4. BUILDINGS AND FACILITIES – CONTAINMENT (p. 7)". The source table has its
# own page column, so the suffix is stripped for display rather than shown twice.
_SECTION_PAGE_SUFFIX = re.compile(r"\s*\((?:pp?\.)\s*[\d\s\-–]+\)\s*$", re.IGNORECASE)


def clean_section(section: Any) -> str | None:
    if not section:
        return None
    text = _SECTION_PAGE_SUFFIX.sub("", str(section)).strip()
    return text or None


def as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── Page ranges ───────────────────────────────────────────────────────────────
def format_pages(page_start: Any, page_end: Any = None) -> str:
    """
    Table form.

        12        -> "12"
        12, 14    -> "12-14"
        None      -> "N/A"
    """
    start, end = as_int(page_start), as_int(page_end)

    if start is None and end is None:
        return "N/A"
    if start is None:
        return str(end)
    if end is None or end == start:
        return str(start)
    if end < start:
        start, end = end, start
    return f"{start}-{end}"


def format_pages_inline(page_start: Any, page_end: Any = None) -> str:
    """
    Prose form, for inside a citation bracket.

        12      -> "p. 12"
        12, 14  -> "pp. 12-14"
        None    -> ""
    """
    start, end = as_int(page_start), as_int(page_end)

    if start is None and end is None:
        return ""
    if start is None:
        return f"p. {end}"
    if end is None or end == start:
        return f"p. {start}"
    if end < start:
        start, end = end, start
    return f"pp. {start}-{end}"


def merge_page_ranges(ranges: Iterable[tuple[int | None, int | None]]) -> list[tuple[int, int]]:
    """
    Collapse the page spans of several chunks into the fewest ranges.

    Chunks on pages 11, 12 and 13 of one document become a single "11-13" row
    rather than three near-identical rows. Non-adjacent spans stay separate: a
    document that answers a question on page 4 and again on page 40 is telling
    the reader two different places to look, and flattening that to "4-40" would
    be a false statement about where the evidence is.
    """
    spans: list[tuple[int, int]] = []
    for start, end in ranges:
        start_i, end_i = as_int(start), as_int(end)
        if start_i is None and end_i is None:
            continue
        if start_i is None:
            start_i = end_i
        if end_i is None:
            end_i = start_i
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        spans.append((start_i, end_i))

    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        # <= last_end + 1 so that 11-12 and 13 join: they are contiguous pages,
        # not two separate places in the document.
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def format_page_ranges(spans: list[tuple[int, int]]) -> str:
    """"11-13", or "4, 11-13" when the evidence sits in two places."""
    if not spans:
        return "N/A"
    return ", ".join(
        str(start) if start == end else f"{start}-{end}" for start, end in spans
    )


# ── Citation keys ─────────────────────────────────────────────────────────────
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")

# Roles that genuinely distinguish two documents sharing one reference code:
# ICH Q7 the guideline vs ICH Q7 the Q&A; PE 009-17 Introduction vs Annexes.
_ROLE_SUFFIX = {
    "q&a": "QA",
    "qa": "QA",
    "annex": "ANNEX",
    "annexes": "ANNEXES",
    "introduction": "INTRO",
}


def normalise_key(text: str, max_length: int = 44) -> str:
    key = _NON_ALNUM.sub("_", str(text)).strip("_").upper()
    key = re.sub(r"_+", "_", key)
    if len(key) <= max_length:
        return key
    # Trim on a word boundary so the key stays readable.
    trimmed = key[:max_length]
    if "_" in trimmed:
        trimmed = trimmed[: trimmed.rfind("_")]
    return trimmed.strip("_") or key[:max_length]


def citation_key(payload: dict) -> str:
    """
    A short, stable handle for one document.

    Derived from real stored metadata — the reference code where the registry
    has one, otherwise a trimmed document_id — never composed by the model, and
    always reconciled back to a real document_id before anything is returned.
    The registry's own ids would be correct here but unreadable: citing
    "[EUDRALEX_01_GUIDELINES_ON_GOOD_MANUFACTURING_PRACTICE_SPECIFIC_TO_ADVANCED_THERAPY_MEDICINAL_PRODUCTS, p. 71]"
    mid-sentence helps nobody.
    """
    reference = (payload.get("reference_code") or "").strip()
    role = (payload.get("content_role") or "").strip().lower()

    if reference:
        key = normalise_key(reference)
        suffix = _ROLE_SUFFIX.get(role)
        if suffix and not key.endswith(suffix):
            key = f"{key}_{suffix}"
        return key

    return normalise_key(payload.get("document_id") or payload.get("document_key") or "DOC")


def assign_citation_keys(chunks: list) -> dict[str, str]:
    """
    Map document_id -> citation key over one result set, keeping keys unique.

    Two different documents can normalise to the same key (an EudraLex title
    trimmed at 44 characters, say). Within a single answer that would make the
    citation ambiguous, so the loser falls back to its full document_id — still
    correct, just longer.
    """
    keys: dict[str, str] = {}
    taken: dict[str, str] = {}

    for chunk in chunks:
        payload = chunk.payload if hasattr(chunk, "payload") else chunk
        document_id = str(payload.get("document_id") or payload.get("document_key") or "")
        if not document_id or document_id in keys:
            continue

        key = citation_key(payload)
        if taken.get(key, document_id) != document_id:
            key = normalise_key(document_id, max_length=200)

        keys[document_id] = key
        taken[key] = document_id

    return keys


# ── Descriptions ──────────────────────────────────────────────────────────────
_WEIGHT_WORDS = {
    "regulation": "binding regulation",
    "official_guideline": "official guideline",
    "official_qa": "official questions-and-answers document",
    "inspection_guidance": "inspection guidance",
    "supporting_guidance": "supporting guidance",
}


def build_description(payload: dict) -> str:
    """
    A one-sentence description, composed from stored metadata.

    Assembled rather than generated: the registry already records authority,
    document type, regulatory weight and topic, so asking a model to describe
    the document each time would spend tokens to produce something less
    reliable and different on every run.
    """
    authority = (payload.get("authority") or "").strip()
    weight = _WEIGHT_WORDS.get((payload.get("regulatory_weight") or "").strip())
    document_type = (payload.get("document_type") or "").strip().replace("_", " ")
    title = (payload.get("document_title") or "").strip()
    topic = (payload.get("topic") or "").strip()
    reference = (payload.get("reference_code") or "").strip()

    kind = weight or (f"{document_type} document" if document_type else "document")

    parts = [f"{authority} {kind}".strip() if authority else kind.capitalize()]

    if title:
        parts.append(f"covering {title.rstrip('.')}")
    elif topic:
        parts.append(f"covering {topic}")

    if reference and reference.lower() not in title.lower():
        parts.append(f"({reference})")

    sentence = " ".join(parts).strip()
    sentence = sentence[0].upper() + sentence[1:] if sentence else "Regulatory document."
    return sentence if sentence.endswith(".") else sentence + "."


def official_url(payload: dict) -> str | None:
    """document_url, then source_page_url — the order the brief specifies."""
    for key in ("document_url", "source_page_url", "citation_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
