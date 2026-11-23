"""
text_utils.py — the text cleaning every extractor shares.

Whitespace normalisation, line cleaning and letterhead detection were written
twice: once for the knowledge-base extractors and once for the upload readers,
with identical implementations. They live here now, so a change to what counts
as boilerplate reaches both.
"""

from __future__ import annotations

import re

_WS_RE  = re.compile("[ \t\u00a0]+")
_TAG_RE = re.compile(r"<[^>]{1,40}>")


def clean_line(line: str) -> str:
    """
    Reduce one extracted line to plain prose.

    pymupdf4llm renders cover pages as markdown tables and leaves inline HTML
    in place, so a raw line often arrives as
    "| Annex 19 <br> Reference and Retention Samples ||".
    """
    line = _TAG_RE.sub(" ", line)
    line = line.replace("|", " ").replace("*", " ")
    return _WS_RE.sub(" ", line).strip(" #_-—:")


def normalise_text(text: str) -> str:
    """Collapse runs of blank lines and trailing spaces; keep paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS_RE.sub(" ", line).rstrip() for line in text.split("\n")]

    out: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks < 2:
                out.append("")
    return "\n".join(out).strip()


# Institutional letterhead. Both EudraLex and ICH open with several lines of it
# before any content, and it is worthless as retrievable text.
BOILERPLATE = re.compile(
    r"european commission|enterprise and industry|health and consumers"
    r"|directorate[-\s]?general|consumer goods|^pharmaceuticals$|brussels"
    r"|eudralex|rules governing medicinal products|volume\s*4"
    r"|public health and risk assessment|international council for harmonisation"
    r"|ich harmonised|harmonisation of technical requirements"
    r"|^final version|document history|table of contents|^ref\.?\s*ares"
    r"|^step\s*\d|current step|^\s*legal notice"
    r"|^eu guidelines?\b|medicinal products for human and veterinary use"
    r"|^good manufacturing practice$|^annex\s*\d+$|^chapter\s*\d+$",
    re.IGNORECASE,
)

_TABLE_NOISE = re.compile(r"^[\s|:-]*$")


def strip_front_matter(records: list[dict]) -> list[dict]:
    """
    Drop the leading run of letterhead from a paginated document's first page.

    Only a leading run, and only on page one: the moment a line of real content
    appears, stripping stops. A cover page that is boilerplate all the way down
    is kept rather than emptied out.
    """
    if not records:
        return records

    first = records[0]
    if first.get("atomic"):
        return records

    lines = first["text"].splitlines()
    cut = 0
    for index, line in enumerate(lines):
        cleaned = clean_line(line)
        if not cleaned or _TABLE_NOISE.match(line) or BOILERPLATE.search(cleaned):
            cut = index + 1
            continue
        break

    if cut and cut < len(lines):
        remainder = "\n".join(lines[cut:]).strip()
        if remainder:
            return [{**first, "text": remainder}] + records[1:]

    return records
