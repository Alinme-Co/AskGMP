"""
html.py — HTML into one record per heading section.

Sectioning on headings rather than returning the page as one blob is what keeps
a citation meaningful: eCFR pages carry every section of a CFR part in one
file, and a single-blob extraction would cite "21 CFR Part 211" for a hit that
actually lives in § 211.192.

Two things this learned the hard way against the real corpus, both still load
bearing: find `<main>`/`<article>` before sectioning, or a page's navigation
ends up indexed alongside its content; and never decompose an element that
holds most of the page, because a layout wrapper whose class happens to contain
"header" will otherwise take the whole document with it.

Identical for the knowledge base and for uploads — an HTML file has no pictures
worth OCR-ing and no pagination, so the profile makes no difference here.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.documents.models import KIND_TEXT, drop_empty, record
from app.documents.text_utils import _WS_RE, normalise_text

SUFFIXES = (".html", ".htm", ".xhtml")

_HTML_NOISE_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer", "aside",
    "form", "button", "svg", "iframe", "template",
]
_HTML_NOISE_ATTR = re.compile(
    r"nav|menu|breadcrumb|banner|cookie|sidebar|footer|header|skip-link|toolbar|search",
    re.IGNORECASE,
)
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

# Containers that, when a page has one, hold the document and not the website.
_MAIN_SELECTORS = (
    "main", "[role=main]", "#main-content", "#main", "article",
    "#document1", "#content", ".content",
)


def _main_container(soup, body):
    """
    The subtree holding the actual document, or the body if there isn't one.

    EMA's pages wrap ~2000 characters of guideline inside ~5000 characters of
    mega-menu, language picker and footer. Sectioning the whole body puts the
    site navigation into the index; finding <main> first leaves it out.
    """
    whole = len(body.get_text(" ", strip=True))
    best, best_size = body, 0

    for selector in _MAIN_SELECTORS:
        try:
            candidates = soup.select(selector)
        except Exception:                        # noqa: BLE001 - odd selector support
            continue
        for element in candidates:
            size = len(element.get_text(" ", strip=True))
            if size > best_size and (size >= 500 or size >= whole * 0.3):
                best, best_size = element, size

    return best


def _strip_noise(container) -> None:
    """
    Drop navigation-ish elements — but never one large enough to be the page.

    Matching "header|nav|search|..." as a substring of a class name catches real
    chrome, and also catches the layout wrapper whose class happens to contain
    "header". decompose() takes the whole subtree with it, so one such match on
    an ancestor silently empties the document: on EMA's ICH Q10 page it took the
    text from 4803 characters to 94. The size guard is what makes the rule safe
    — chrome is a small fraction of a page; the document is not.
    """
    whole = len(container.get_text(" ", strip=True)) or 1

    for attribute in ("class", "id"):
        for element in list(container.find_all(attrs={attribute: _HTML_NOISE_ATTR})):
            if element.decomposed:
                continue
            if len(element.get_text(" ", strip=True)) < whole * 0.4:
                element.decompose()

    for element in list(container.find_all(
        attrs={"role": re.compile("navigation|banner|search", re.I)}
    )):
        if not element.decomposed and len(element.get_text(" ", strip=True)) < whole * 0.4:
            element.decompose()


def extract(path: str, **_) -> tuple[list[dict], str]:
    from bs4 import BeautifulSoup

    markup = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(markup, "html.parser")

    for tag in soup(_HTML_NOISE_TAGS):
        tag.decompose()

    body = soup.body or soup
    container = _main_container(soup, body)
    _strip_noise(container)
    body = container

    records: list[dict] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = normalise_text("\n\n".join(buffer))
        if text:
            headed = f"{section}\n\n{text}" if section else text
            records.append(record(
                headed,
                kind=KIND_TEXT,
                section=section,
                locator=section,
                # Heading sections read as flowing prose, so the chunker may
                # merge two of them when they share a section label.
                atomic=False,
                heading=section,
            ))
        buffer.clear()

    for element in body.find_all([*_HEADING_TAGS, "p", "li", "td", "th", "pre", "blockquote"]):
        text = _WS_RE.sub(" ", element.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if element.name in _HEADING_TAGS:
            flush()
            section = text[:160]
        else:
            buffer.append(text)

    flush()

    if not records:
        text = normalise_text(body.get_text("\n", strip=True))
        if text:
            records = [record(text, kind=KIND_TEXT)]

    return drop_empty(records), "extracted"
