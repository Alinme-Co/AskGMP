"""
citations.py — Reconcile what the model claimed against what was retrieved.

The model returns two things that can be wrong in different ways: structured
citation objects, and inline `[KEY, p. N]` brackets inside the prose. Both are
checked here against the retrieved chunks, and anything that cannot be traced
back to a real chunk is removed.

Three failure modes are handled:

  * an unknown key — the model cited a document that was not retrieved. The
    bracket is stripped from the prose and the key reported in
    `rejected_citations`, so the failure is visible rather than silent.

  * a page the document does not have on this retrieval — the model read
    "p. 7" and wrote "p. 9". The page is corrected to the nearest range the
    document actually contributed, rather than the citation being dropped: the
    document is right, only the locator is off.

  * a page cited for a document that has no page metadata at all (a JSON Q&A
    collection, an HTML page). The page is removed and the bare key kept.

Source rows are then built from the *payloads*, never from the model's output,
and chunks of one document are aggregated into a single row with merged page
ranges.
"""

from __future__ import annotations

import re
from typing import Iterable

from app.chat.formatting import (
    as_int,
    build_description,
    clean_section,
    format_page_ranges,
    format_pages,
    format_pages_inline,
    merge_page_ranges,
    official_url,
)
from app.chat.schemas import AskGMPSource, LLMCitation

# [KEY], [KEY, p. 12], [KEY, pp. 12-14], and multi-citation brackets
# [KEY_A, pp. 5-6; KEY_B, p. 8].
_CITATION_BRACKET = re.compile(r"\[([^\[\]]{1,400})\]")

# The key is required to be upper-case and at least three characters, which is
# what every generated citation key looks like. That is the discriminator
# between a citation and ordinary prose in brackets — "[or their deputy]" must
# survive untouched, and would otherwise parse as key "or" with a trailing
# qualifier.
_ONE_CITATION = re.compile(
    r"^\s*(?P<key>[A-Z0-9][A-Z0-9_\-./()&]{2,})"
    r"(?:\s*,\s*(?:"
    r"(?:pp?\.)\s*(?P<start>\d+)(?:\s*[-–]\s*(?P<end>\d+))?"
    r"|(?P<tail>[^;\]]{1,80})"
    r"))?\s*$"
)

# Documents without pages get cited by section instead — "[21_CFR_PART_211,
# Section 211.100]". That form still has to be checked, or an invented key
# wearing a section label would pass straight through.
_SECTION_TAIL = re.compile(
    r"^(section|sect\.?|§+|art\.?|article|chapter|chap\.?|annex|para\.?|paragraph|clause)\b",
    re.IGNORECASE,
)
_TAIL_TOKEN = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")


def _pages_for(document_id: str, chunks: list) -> list[tuple[int, int]]:
    """Every page span this document actually contributed to the context."""
    spans = []
    for chunk in chunks:
        if str(chunk.payload.get("document_id") or "") != document_id:
            continue
        start = as_int(chunk.payload.get("page"))
        end = as_int(chunk.payload.get("page_end"))
        if start is not None or end is not None:
            spans.append((start, end))
    return merge_page_ranges(spans)


def _section_is_grounded(tail: str, document_id: str, chunks: list) -> bool:
    """
    Is "Section 211.100" something this document's retrieved chunks actually say?

    Checked, not trusted. A section reference is a locator like a page number,
    and an invented one sends the reader to a part of the regulation that does
    not discuss what the answer claims. The test is deliberately simple: the
    tail must look like a section reference, and its identifying token must
    appear in the document's retrieved section labels or text.
    """
    if not _SECTION_TAIL.match(tail):
        return False

    haystack = " ".join(
        f"{chunk.payload.get('section') or ''} {chunk.text or ''}"
        for chunk in chunks
        if str(chunk.payload.get("document_id") or "") == document_id
    ).lower()

    tokens = [
        token for token in _TAIL_TOKEN.findall(tail)
        if not _SECTION_TAIL.match(token)
    ]
    if not tokens:
        return False

    return all(token.lower() in haystack for token in tokens)


def _closest_span(page: int, spans: list[tuple[int, int]]) -> tuple[int, int]:
    """The retrieved span nearest a page the model invented."""
    for start, end in spans:
        if start <= page <= end:
            return start, end
    return min(spans, key=lambda span: min(abs(span[0] - page), abs(span[1] - page)))


def validate_inline_citations(
    answer: str,
    key_to_document: dict[str, str],
    chunks: list,
) -> tuple[str, set[str], list[str]]:
    """
    Clean the prose. Returns (answer, keys actually used, rejected keys).

    Brackets that contain no recognisable citation — ordinary prose in square
    brackets — are left alone.
    """
    used: set[str] = set()
    rejected: list[str] = []
    spans_cache: dict[str, list[tuple[int, int]]] = {}

    def fix_bracket(match: re.Match) -> str:
        body = match.group(1)
        parts = [part for part in re.split(r"\s*;\s*", body) if part.strip()]

        parsed = [_ONE_CITATION.match(part) for part in parts]
        if not parts or any(item is None for item in parsed):
            return match.group(0)          # not a citation; leave untouched

        kept: list[str] = []
        for item in parsed:
            key = item.group("key")
            document_id = key_to_document.get(key)

            if document_id is None:
                if key not in rejected:
                    rejected.append(key)
                continue

            used.add(key)

            start, end = as_int(item.group("start")), as_int(item.group("end"))
            if start is None:
                tail = (item.group("tail") or "").strip()
                if tail and _section_is_grounded(tail, document_id, chunks):
                    kept.append(f"{key}, {tail}")
                else:
                    kept.append(key)
                continue

            spans = spans_cache.setdefault(document_id, _pages_for(document_id, chunks))
            if not spans:
                kept.append(key)           # document has no pages; drop the locator
                continue

            if not any(s <= start <= e for s, e in spans):
                start, end = _closest_span(start, spans)
            elif end is not None and not any(s <= end <= e for s, e in spans):
                end = None

            kept.append(f"{key}, {format_pages_inline(start, end)}")

        if not kept:
            return ""                      # every citation in the bracket was bogus
        return "[" + "; ".join(kept) + "]"

    cleaned = _CITATION_BRACKET.sub(fix_bracket, answer)
    # Tidy the punctuation a removed bracket leaves behind.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)

    return cleaned.strip(), used, rejected


def validate_structured_citations(
    citations: Iterable[LLMCitation],
    key_to_document: dict[str, str],
) -> tuple[list[LLMCitation], list[str]]:
    """Drop structured citations whose key was never retrieved."""
    kept, rejected = [], []
    for citation in citations:
        if citation.document_id in key_to_document:
            kept.append(citation)
        elif citation.document_id not in rejected:
            rejected.append(citation.document_id)
    return kept, rejected


def cited_page_spans(
    answer: str, citations: Iterable[LLMCitation]
) -> dict[str, list[tuple[int, int]]]:
    """
    key -> the page spans the validated answer actually cites.

    Read from the cleaned prose and the kept structured citations, so it only
    ever reports pages that already survived validation.
    """
    spans: dict[str, list[tuple[int, int]]] = {}

    def add(key: str, start, end) -> None:
        start, end = as_int(start), as_int(end)
        if start is not None:
            spans.setdefault(key, []).append((start, end if end is not None else start))

    for match in _CITATION_BRACKET.finditer(answer):
        for part in re.split(r"\s*;\s*", match.group(1)):
            item = _ONE_CITATION.match(part)
            if item:
                add(item.group("key"), item.group("start"), item.group("end"))
    for citation in citations:
        add(citation.document_id, citation.page_start, citation.page_end)
    return spans


def build_sources(
    chunks: list,
    key_to_document: dict[str, str],
    cited_keys: set[str] | None = None,
) -> list[AskGMPSource]:
    """
    One row per document, with that document's chunks aggregated.

    Five chunks from one guideline are one row, not five. Their pages are
    merged into the fewest contiguous ranges, and the best score across the
    document's chunks becomes the row's relevance. Rows the answer actually
    cited sort first — the rest are still returned, because "here is what I
    looked at" is useful even when a passage did not make it into the prose.
    """
    cited_keys = cited_keys or set()
    document_to_key = {value: key for key, value in key_to_document.items()}

    grouped: dict[str, list] = {}
    for chunk in chunks:
        document_id = str(chunk.payload.get("document_id") or "")
        if document_id:
            grouped.setdefault(document_id, []).append(chunk)

    sources: list[AskGMPSource] = []

    for document_id, document_chunks in grouped.items():
        best = max(document_chunks, key=lambda item: item.score)
        payload = best.payload

        spans = _pages_for(document_id, document_chunks)
        page_start = spans[0][0] if spans else None
        page_end = spans[-1][1] if spans else None

        sections = []
        for chunk in sorted(document_chunks, key=lambda item: -item.score):
            section = clean_section(chunk.payload.get("section"))
            if section and section not in sections:
                sections.append(section)

        key = document_to_key.get(document_id, document_id)

        sources.append(AskGMPSource(
            citation_key      = key,
            document_id       = document_id,
            document_title    = str(payload.get("document_title") or document_id),
            description       = build_description(payload),
            origin            = str(payload.get("origin") or "knowledge_base"),
            authority         = payload.get("authority"),
            jurisdiction      = payload.get("jurisdiction"),
            document_type     = payload.get("document_type"),
            reference_code    = payload.get("reference_code"),
            topic             = payload.get("topic"),
            status            = payload.get("status"),
            regulatory_weight = payload.get("regulatory_weight"),
            page_start        = page_start,
            page_end          = page_end,
            pages             = format_page_ranges(spans) if spans else format_pages(None),
            page_ranges       = [
                str(start) if start == end else f"{start}-{end}" for start, end in spans
            ],
            section           = "; ".join(sections[:2]) or None,
            document_url      = payload.get("document_url"),
            source_page_url   = payload.get("source_page_url"),
            official_url      = official_url(payload),
            source_file       = payload.get("source_file"),
            relevance_score   = round(max(item.score for item in document_chunks), 4),
            cited             = key in cited_keys,
        ))

    sources.sort(key=lambda source: (not source.cited, -(source.relevance_score or 0.0)))
    return sources
