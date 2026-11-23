"""
summarise.py — what an uploaded document says, when nobody has asked anything.

Someone who uploads a file before typing a question wants to know the upload
worked and what the system made of it. A record count does not answer that; a
summary does.

This is deliberately a different job from answering. There is no retrieval and
no knowledge base involved: the document is summarised on its own terms, with
no assessment of whether it complies with anything. A user's SOP is evidence of
what their site does, and judging it against guidance nobody asked about would
be both unhelpful and, when wrong, alarming.

The grounding rules still apply. Key points carry the locator the extractor
produced — "Sheet 'Specs', row 4", "Slide 3", "p. 2" — so every claim in the
summary can be traced back into the file.

It lives with the uploads because that is what it summarises, and it borrows
`app.chat.prompts` for the source-block rendering and `app.chat.schemas` for the
structured-output models. Those are the same two things the answering path
needs, and a second copy of either would be a second thing to keep correct.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app import config
from app.chat import prompts
from app.chat.schemas import DocumentSummary, LLMDocumentSummary, SummaryPoint
from app.infrastructure.openai_client import get_chat_client

logger = logging.getLogger(__name__)


class DocumentSummariser:
    """Summarises one uploaded document from its own chunks."""

    def __init__(self, upload_index=None, openai_client=None, model: str | None = None):
        self._uploads = upload_index
        self._openai = openai_client
        self.model = model or config.CHAT_MODEL

    @property
    def uploads(self):
        if self._uploads is None:
            from app.uploads.search import shared_index

            self._uploads = shared_index()
        return self._uploads

    @property
    def openai(self):
        if self._openai is None:
            self._openai = get_chat_client()
        return self._openai

    def summarise(self, file_hash: str, filename: str | None = None) -> DocumentSummary:
        chunks = self.uploads.all_chunks(file_hash)
        document = self.uploads.get(file_hash)
        name = filename or (document.filename if document else file_hash[:12])

        if not chunks:
            return DocumentSummary(
                file_hash=file_hash,
                filename=name,
                title=Path(name).stem.replace("_", " "),
                document_kind="Unclear",
                summary="",
                chunk_count=0,
                error="No readable text was found in this document.",
            )

        # Take the opening of a long document rather than the middle: a
        # controlled document states its purpose, scope and responsibilities
        # first, which is exactly what a summary needs.
        truncated = len(chunks) > config.MAX_SUMMARY_CHUNKS
        selected = chunks[: config.MAX_SUMMARY_CHUNKS]

        keys = {chunk.payload["document_id"]: _short_key(name) for chunk in selected}
        context = prompts.build_context(selected, keys)

        try:
            completion = self.openai.chat.completions.parse(
                model=self.model,
                temperature=config.TEMPERATURE,
                messages=[
                    {"role": "system", "content": prompts.SUMMARY_PROMPT},
                    {"role": "user",
                     "content": prompts.build_summary_message(name, context, truncated)},
                ],
                response_format=LLMDocumentSummary,
            )
            parsed = completion.choices[0].message.parsed
        except Exception as exc:                  # noqa: BLE001
            logger.exception("Could not summarise %s", name)
            return DocumentSummary(
                file_hash=file_hash,
                filename=name,
                title=Path(name).stem.replace("_", " "),
                document_kind="Unclear",
                summary="",
                chunk_count=len(chunks),
                error=f"{type(exc).__name__}: {exc}",
            )

        if parsed is None:
            return DocumentSummary(
                file_hash=file_hash, filename=name,
                title=Path(name).stem.replace("_", " "),
                document_kind="Unclear", summary="",
                chunk_count=len(chunks),
                error="The summariser returned nothing usable.",
            )

        return DocumentSummary(
            file_hash=file_hash,
            filename=name,
            title=parsed.title,
            document_kind=parsed.document_kind,
            summary=parsed.summary,
            key_points=_validate_points(parsed.key_points, selected),
            chunk_count=len(chunks),
            truncated=truncated,
        )


def _short_key(filename: str) -> str:
    """A readable handle for the source blocks; not used for citation."""
    return Path(filename).stem.replace(" ", "_").upper()[:40] or "DOCUMENT"


def _validate_points(points, chunks) -> list[SummaryPoint]:
    """
    Keep a point's locator only if the document really has it.

    Same rule as citation validation, for the same reason: a locator is a
    promise that the reader can go and find the statement, and a made-up one
    sends them to a page that says something else. The point survives; only the
    unverifiable locator is dropped.
    """
    known: set[str] = set()
    for chunk in chunks:
        payload = chunk.payload
        for value in (payload.get("locator"), payload.get("section")):
            if value:
                known.add(str(value).strip().lower())
        page = payload.get("page")
        if page is not None:
            known.add(f"p. {page}")
            known.add(str(page))

    validated: list[SummaryPoint] = []
    for point in points:
        locator = (point.locator or "").strip()
        if locator and locator.lower() not in known:
            # Allow "p. 4" against a chunk spanning 3-6.
            if not _page_is_covered(locator, chunks):
                locator = ""
        validated.append(SummaryPoint(point=point.point, locator=locator or None))

    return validated


def _page_is_covered(locator: str, chunks) -> bool:
    import re

    match = re.search(r"(\d+)", locator)
    if not match:
        return False
    page = int(match.group(1))

    for chunk in chunks:
        start = chunk.payload.get("page")
        end = chunk.payload.get("page_end") or start
        if start is not None and int(start) <= page <= int(end):
            return True
    return False
