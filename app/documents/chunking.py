"""
chunking.py — Group extracted records into chunks that a retriever can cite.

One chunker, both workflows. The knowledge-base indexer chunks a EudraLex annex
with it and app/uploads/search.py chunks a user's SOP with it, so a passage from
either is the same size and carries the same locators — which is what lets an
answer weigh the two side by side.

The indexer this replaced used an 8000-token budget, which was a mistake worth
naming: an 8000-token chunk is roughly thirty thousand characters, so the two
sentences that actually matched the query sit somewhere inside a wall of text,
and the citation can only say "somewhere in this document". AskGMP's default is
1000 tokens, and the rules below all pull the same way — toward a chunk that is
about one thing, so the answer can say *which* thing.

Three rules:

  1. An atomic record (one Q&A pair, one CFR section, one slide) is never
     merged with a neighbour. A short Q&A stays one chunk even though it uses a
     fraction of the budget; filling the rest with the *next* Q&A would only
     make both harder to retrieve.

  2. Non-atomic records (PDF pages, HTML heading sections) are merged with
     following records only while they belong together — same section, and
     consecutive pages.

  3. Anything over budget is split on structure first (markdown headings, then
     paragraphs, then sentences) and only falls back to splitting mid-prose
     when a single paragraph is itself too long.
"""

from __future__ import annotations

import re

from app import config

# The budget, in tokens, and the fragment floor. Both live in app/config.py
# with the rest of the tunables and are re-exported here because the chunker is
# where they read naturally at a call site.
MAX_CHUNK_TOKENS = config.MAX_CHUNK_TOKENS
MIN_CHUNK_TOKENS = config.MIN_CHUNK_TOKENS

_HEADING_RE    = re.compile(r"^#{1,6}\s+(.{3,160})$", re.MULTILINE)
_MD_SPLIT_RE   = re.compile(r"^(?=#{1,6}\s)", re.MULTILINE)
_SENTENCE_RE   = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


# ── Token counting ────────────────────────────────────────────────────────────
_TOKENIZER = None
_TOKENIZER_READY = False


def _tokenizer():
    """
    tiktoken if it is installed, otherwise None.

    Lazy and optional so that chunking — and therefore the whole --dry-run path
    and the unit tests — works in an environment that has not installed the
    embedding stack. token_count() falls back to a character estimate, which is
    close enough for boundary decisions; the budget itself is what the
    fingerprint records, not the counter.
    """
    global _TOKENIZER, _TOKENIZER_READY
    if not _TOKENIZER_READY:
        _TOKENIZER_READY = True
        try:
            import tiktoken

            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:                        # noqa: BLE001 - optional dependency
            _TOKENIZER = None
    return _TOKENIZER


def tokenizer_available() -> bool:
    return _tokenizer() is not None


def token_count(text: str) -> int:
    encoder = _tokenizer()
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    # ~4 characters per token for English prose; deliberately conservative so
    # the estimate errs toward smaller chunks rather than over-budget ones.
    return max(1, (len(text) + 3) // 4)


# ── Splitting ─────────────────────────────────────────────────────────────────
def split_text(text: str, budget: int = MAX_CHUNK_TOKENS) -> list[str]:
    """
    Break over-budget text along the strongest structural boundary available.

    Headings first, then blank-line paragraphs, then sentences, and only then
    words. Each level packs greedily, so a run of small paragraphs becomes one
    chunk rather than many.
    """
    text = text.strip()
    if not text:
        return []
    if token_count(text) <= budget:
        return [text]

    # Headings are a boundary, not a hint. Two heading sections that happen to
    # fit in one budget are still two subjects, and gluing them together to use
    # the space up is precisely the behaviour that made the old 8000-token
    # chunks impossible to cite. So headings split without packing; paragraphs
    # and sentences, which live *inside* one section, do pack.
    headings = _by_headings(text)
    if len(headings) > 1:
        parts: list[str] = []
        for unit in headings:
            parts.extend(split_text(unit, budget))
        return parts

    for units in (_by_paragraphs(text), _by_sentences(text)):
        if len(units) > 1:
            return _pack(units, budget)

    return _by_words(text, budget)


def _by_headings(text: str) -> list[str]:
    return [part.strip() for part in _MD_SPLIT_RE.split(text) if part.strip()]


def _by_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _by_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def _by_words(text: str, budget: int) -> list[str]:
    """
    Last resort: halve on word boundaries until each half fits.

    Terminates because every half is strictly shorter, and a single
    unsplittable word is returned as-is rather than recursing forever.
    """
    if token_count(text) <= budget:
        return [text]

    words = text.split()
    if len(words) < 2:
        return [text]

    middle = len(words) // 2
    return (
        _by_words(" ".join(words[:middle]), budget)
        + _by_words(" ".join(words[middle:]), budget)
    )


def _pack(units: list[str], budget: int) -> list[str]:
    """Greedily fill chunks from consecutive units, recursing into oversized ones."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffered = 0

    for unit in units:
        size = token_count(unit)

        if size > budget:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffered = [], 0
            chunks.extend(split_text(unit, budget))
            continue

        if buffer and buffered + size > budget:
            chunks.append("\n\n".join(buffer))
            buffer, buffered = [], 0

        buffer.append(unit)
        buffered += size

    if buffer:
        chunks.append("\n\n".join(buffer))

    return chunks


# ── Section labels ────────────────────────────────────────────────────────────
def section_label(
    text: str,
    section: str | None,
    page_start: int | None,
    page_end: int | None,
) -> str:
    """
    The short human label a citation shows: "§ 211.192 (p. 4)".

    The record's own section wins, because extraction knew the real heading. A
    markdown heading inside the text is the next best thing, and a page range is
    the honest last resort.
    """
    from app.documents.text_utils import BOILERPLATE, clean_line

    # clean_line strips the inline HTML pymupdf4llm leaves behind — without it
    # labels arrive as "<u>Guidelines on Good Manufacturing Practice</u>".
    label = clean_line(section or "")

    if not label:
        for match in _HEADING_RE.finditer(text):
            heading = clean_line(match.group(1))
            if heading and not BOILERPLATE.search(heading):
                label = heading
                break

    pages = page_range(page_start, page_end)

    if label and pages:
        return f"{label[:110]} ({pages})"
    if label:
        return label[:140]
    return pages or "Full document"


def page_range(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return f"p. {page_start}"
    return f"pp. {page_start}-{page_end}"


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_records(records: list[dict], budget: int = MAX_CHUNK_TOKENS) -> list[dict]:
    """
    Records -> chunks.

    Each chunk carries text, its page range, a section label and the `meta` of
    the record(s) it came from, so Q&A metadata and per-record source URLs
    survive all the way into the payload.

    Grouping is done over records first and only materialised at the end, so
    that _absorb_fragments can still rejoin a stranded fragment to its
    neighbour without having to unpick joined text.
    """
    groups: list[list[dict]] = []
    buffer: list[dict] = []
    buffered = 0

    def flush() -> None:
        nonlocal buffer, buffered
        if buffer:
            groups.append(buffer)
        buffer, buffered = [], 0

    for record in records:
        text = (record.get("text") or "").strip()
        if not text:
            continue

        size = token_count(text)
        atomic = bool(record.get("atomic"))

        if atomic or size > budget:
            # A scrap sitting in the buffer — the EudraLex glossary's opening
            # page is the single word "GLOSSARY" — would be flushed as a chunk
            # of its own here, and a four-token chunk can never answer
            # anything. Fold it into the text about to be split instead, where
            # it reads as the heading it actually is.
            prefix: list[dict] = []
            if (
                buffer
                and not atomic
                and buffered < MIN_CHUNK_TOKENS
                and _continuous(buffer[-1], record)
            ):
                prefix, buffer, buffered = buffer, [], 0

            flush()

            combined = "\n\n".join(
                [item["text"].strip() for item in prefix] + [text]
            ) if prefix else text

            parts = (
                split_text(combined, budget)
                if token_count(combined) > budget else [combined]
            )
            for position, part in enumerate(parts):
                synthetic = {**record, "text": part}
                if position == 0 and prefix and prefix[0].get("page") is not None:
                    # The passage starts on the fragment's page, so that is
                    # where a citation should point.
                    synthetic["page"] = prefix[0]["page"]
                groups.append([synthetic])
            continue

        if buffer and (buffered + size > budget or not _continuous(buffer[-1], record)):
            flush()

        buffer.append(record)
        buffered += size

    flush()

    groups = _absorb_fragments(groups, budget)

    chunks = [_merge(group) for group in groups]
    return [chunk for chunk in chunks if chunk["text"].strip()]


def _absorb_fragments(groups: list[list[dict]], budget: int) -> list[list[dict]]:
    """
    Rejoin a stranded scrap to the passage it belongs to.

    The EudraLex glossary opens with a page holding the single word "GLOSSARY";
    the page after it is over budget, which forces a flush and leaves that word
    as a chunk of its own. A four-token chunk is pure noise in a retrieval set —
    it can never answer anything, and it competes for a slot.

    Only non-atomic, page-continuous neighbours are considered, and only when
    the result still fits: a short Q&A is a complete thought, not a fragment,
    and must never be absorbed into its neighbour.
    """
    if len(groups) < 2:
        return groups

    def size(group: list[dict]) -> int:
        return sum(token_count(record["text"]) for record in group)

    def atomic(group: list[dict]) -> bool:
        return any(record.get("atomic") for record in group)

    merged: list[list[dict]] = []

    for group in groups:
        if (
            merged
            and not atomic(group)
            and not atomic(merged[-1])
            and size(group) < MIN_CHUNK_TOKENS
            and size(merged[-1]) + size(group) <= budget
            and _continuous(merged[-1][-1], group[0])
        ):
            merged[-1].extend(group)
            continue
        merged.append(group)

    return merged


# Pages that extract to nothing — blank versos, separator sheets, a scanned
# page with no text layer — are dropped during extraction, so consecutive
# records can be two page numbers apart while still being one passage. The
# EudraLex glossary jumps 1 -> 3 for exactly this reason. Tolerating a single
# missing page keeps that passage together; a wider gap is treated as a break.
MAX_PAGE_GAP = 2


def _continuous(previous: dict, current: dict) -> bool:
    """
    May these two records share a chunk?

    Only when they are genuinely adjacent parts of the same passage: the same
    section (or no section at all), and near-consecutive pages if both are
    paginated. Two different heading sections are two different topics, and
    merging them to use up the budget is exactly the behaviour this rewrite set
    out to remove.
    """
    if (previous.get("section") or None) != (current.get("section") or None):
        return False

    before, after = previous.get("page"), current.get("page")
    if before is not None and after is not None and after - before > MAX_PAGE_GAP:
        return False

    return True


def _merge(records: list[dict]) -> dict:
    texts = [record["text"].strip() for record in records if record["text"].strip()]
    pages = [record.get("page") for record in records if record.get("page") is not None]

    meta: dict = {}
    for record in records:
        for key, value in (record.get("meta") or {}).items():
            meta.setdefault(key, value)

    section = next(
        (record.get("section") for record in records if record.get("section")), None
    )
    page_start = min(pages) if pages else None
    page_end   = max(pages) if pages else None
    text = "\n\n".join(texts)

    return {
        "text"      : text,
        "page_start": page_start,
        "page_end"  : page_end,
        "section"   : section_label(text, section, page_start, page_end),
        "meta"      : meta,
    }
