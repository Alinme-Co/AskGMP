"""
json.py — structured JSON becomes one record per logical item, never one blob.

The knowledge base holds three shapes, and flattening any of them into a single
document would be a real loss:

  * Q&A collections (EMA GMP/GDP, FDA cGMP) — question, answer, section, and a
    per-record source URL that is the *exact* page to cite.
  * Section collections (21 CFR 210/211) — section number, heading, text, and
    again a per-section eCFR URL.
  * Anything else — rendered as readable key/value prose.

In the first two cases each record is atomic, so one Q&A or one CFR section
stays one chunk and the citation points at it precisely.

(This module is named `json`; `import json` inside it still resolves to the
standard library, because Python 3 imports are absolute.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.documents.models import KIND_QA, KIND_TEXT, record
from app.documents.text_utils import normalise_text

SUFFIXES = (".json",)

QUESTION_KEYS = ("question", "q", "title")
ANSWER_KEYS   = ("answer", "a", "response", "text", "body")
SECTION_KEYS  = ("section", "heading", "chapter", "category", "part_title")
TOPIC_KEYS    = ("topic", "subject", "theme", "source_type")
URL_KEYS      = ("source_url", "source", "url", "link", "href")


def extract(path: str, **_) -> tuple[list[dict], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))

    items = _json_items(data)
    if items is None:
        return (
            [record(normalise_text(_render_json(data)), kind=KIND_TEXT)],
            "extracted (unstructured)",
        )

    records: list[dict] = []
    status = "extracted"

    for position, item in enumerate(items, 1):
        question = _first(item, QUESTION_KEYS)
        section  = _first(item, SECTION_KEYS)
        topic    = _first(item, TOPIC_KEYS)
        url      = _first(item, URL_KEYS)
        kind     = KIND_TEXT

        if question and _first(item, ANSWER_KEYS):
            status = "extracted (q&a)"
            kind = KIND_QA
            answer = _first(item, ANSWER_KEYS) or ""
            label = section or topic or "Question & Answer"
            text = f"# {label}\n\n## Question\n{question}\n\n## Answer\n{answer}"
            meta = {
                "question"  : question,
                "qa_section": section,
                "qa_topic"  : topic,
            }
        else:
            body = _first(item, ("text", "body", "content")) or _render_json(item)
            heading = _first(item, ("heading", "title")) or ""
            number = _first(item, ("section", "id", "number")) or ""
            label = " ".join(part for part in (number, heading) if part).strip()
            section = label or section or topic
            text = f"# {label}\n\n{body}" if label else str(body)
            meta = {"qa_section": section, "qa_topic": topic}

        if url:
            meta["record_url"] = url

        text = normalise_text(text)
        if not text:
            continue

        records.append(record(
            text,
            kind    = kind,
            page    = None,
            section = (section or topic or f"Record {position}"),
            # One Q&A or one CFR section is a complete unit of meaning; merging
            # it with the next one produces a chunk that answers neither well.
            atomic  = True,
            **{key: value for key, value in meta.items() if value},
        ))

    return records, status


def _json_items(data: Any) -> list[dict] | None:
    """The list of record dicts inside a JSON file, or None if there isn't one."""
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        return data

    if isinstance(data, dict):
        for key in ("records", "items", "data", "questions", "qa", "sections", "results"):
            value = data.get(key)
            if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
                return value

    return None


def _first(item: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _render_json(value: Any, depth: int = 0) -> str:
    """Readable prose for JSON that is not a recognised record collection."""
    indent = "  " * depth
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = str(key).replace("_", " ").strip().capitalize()
            if isinstance(item, (dict, list)):
                lines.append(f"{indent}{label}:")
                lines.append(_render_json(item, depth + 1))
            elif item not in (None, ""):
                lines.append(f"{indent}{label}: {item}")
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join(_render_json(item, depth) for item in value)
    return f"{indent}{value}"
