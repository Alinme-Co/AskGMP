"""
service.py — question in, grounded answer plus source table out.

    retrieve → build context → structured LLM call → validate → build sources

The ordering matters at two points. Retrieval happens before any model call, so
a question with no supporting documents costs nothing and returns the no-answer
response directly. And validation happens after generation but before anything
is returned, so a citation the model invented never reaches the caller.

The service returns sources even when it declines to answer. "I could not find
enough information, and here is the closest thing I did find" is a useful
answer for someone trying to locate a document; a bare refusal is not.
"""

from __future__ import annotations

import logging
from typing import Any

from app import config
from app.chat import prompts
from app.chat.citations import (
    build_sources,
    cited_page_spans,
    validate_inline_citations,
    validate_structured_citations,
)
from app.chat.formatting import assign_citation_keys
from app.chat.schemas import AskGMPResponse, LLMAnswer
from app.infrastructure.openai_client import get_chat_client
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

_DEFAULT = object()


class AskGMPService:
    """
    Orchestrates one question.

    Both clients are injectable, so the whole flow — including citation
    validation and source aggregation — is testable without OpenAI or Qdrant.
    """

    def __init__(self, retriever: Retriever | None = None, openai_client=None,
                 model: str | None = None, upload_index=None, core=_DEFAULT):
        self.retriever = retriever or Retriever()
        self._openai = openai_client
        self.model = model or config.CHAT_MODEL
        self._uploads = upload_index
        # The cached core library (CAG). None turns it off: plain RAG.
        if core is _DEFAULT:
            from app.cag.core import CoreLibrary

            core = CoreLibrary() if config.CAG_ENABLED else None
        self.core: Any = core

    @property
    def uploads(self):
        """
        The in-memory index over uploaded documents, built on first use.

        Injected by the API, and imported lazily here only as the fallback — so
        chat depends on no storage implementation at import time, and a caller
        that supplies its own index never loads app.uploads at all.
        """
        if self._uploads is None:
            from app.uploads.search import shared_index

            self._uploads = shared_index()
        return self._uploads

    def _search_documents(
        self, question: str, document_hashes: list[str] | None
    ) -> list:
        """
        Passages from the user's uploads. Never fatal.

        A failure here — an unreadable file, an embedding error — must not lose
        the knowledge-base answer as well, so it degrades to "no uploaded
        context" rather than raising.
        """
        if not document_hashes:
            return []
        try:
            return self.uploads.search(
                question, document_hashes, use_ocr=config.UPLOAD_OCR
            )
        except Exception:                         # noqa: BLE001
            logger.exception("Could not search the uploaded documents")
            return []

    @property
    def openai(self):
        if self._openai is None:
            self._openai = get_chat_client()
        return self._openai

    # -- the one public entry point --
    def answer(
        self,
        question: str,
        effective_only: bool | None = None,
        authority: str | None = None,
        document_hashes: list[str] | None = None,
    ) -> AskGMPResponse:
        question = (question or "").strip()
        if not question:
            return AskGMPResponse(
                answer=config.NO_ANSWER_TEXT,
                sources=[],
                has_sufficient_evidence=False,
                question=question,
            )

        # The user's own documents first, then the corpus. Both searched with
        # the same question, and the results concatenated rather than merged by
        # score: an RRF score from an in-memory index over forty chunks is not
        # comparable with one from Qdrant over 1,844, so ranking them against
        # each other would be arithmetic without meaning. Ordering by origin is
        # honest and puts the user's own context first, where it belongs for a
        # question like "is this a deviation?".
        document_chunks = self._search_documents(question, document_hashes)

        # The core library is skipped under an authority filter: it holds
        # several authorities, and the filter asks for exactly one. RAG then
        # searches the whole corpus, core documents included.
        core_chunks = self.core.chunks() if self.core and not authority else []
        core_ids = set(self.core.document_ids) if core_chunks else set()

        # Retrieval ranks the whole corpus, then core hits are dropped: they are
        # already in the prompt. A question the core answers therefore sends few
        # or no extra chunks, instead of eight long-tail fillers that ranked
        # below the core — which is most of what the cached prefix would cost.
        retrieved = document_chunks + [
            chunk for chunk in self.retriever.retrieve(
                question, effective_only=effective_only, authority=authority
            )
            if chunk.document_id not in core_ids
        ]
        # Core first, so it always wins its citation keys: the cached prefix
        # stays byte-identical whatever else a question retrieves.
        chunks = core_chunks + retrieved

        # Nothing retrieved: answer without spending a model call. There is
        # nothing for the model to be grounded in, so calling it could only
        # produce an ungrounded answer.
        if len(chunks) < config.MIN_RESULTS_FOR_ANSWER:
            return AskGMPResponse(
                answer=config.NO_ANSWER_TEXT,
                sources=[],
                has_sufficient_evidence=False,
                question=question,
            )

        key_to_document = _invert(assign_citation_keys(chunks))

        completion = self._generate(question, core_chunks, retrieved, key_to_document)

        return self._reconcile(question, completion, chunks, key_to_document,
                               core_count=len(core_chunks))

    # -- generation --
    def _generate(self, question: str, core_chunks: list, chunks: list,
                  key_to_document: dict[str, str]) -> LLMAnswer:
        document_to_key = {value: key for key, value in key_to_document.items()}
        context = prompts.build_context(chunks, document_to_key, start=len(core_chunks) + 1)

        # Static first, dynamic last. The core block carries the cache
        # breakpoint, so everything up to it is read from cache on later calls;
        # the retrieved blocks and the question come after it.
        system: list[dict] = [{"type": "text", "text": prompts.SYSTEM_PROMPT}]
        if core_chunks:
            system.append({
                "type": "text",
                "text": prompts.build_core_context(core_chunks, document_to_key),
                "cache_control": config.CAG_CACHE_CONTROL,
            })

        # Any: the SDK's message TypedDicts do not declare cache_control.
        messages: list[Any] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompts.build_user_message(question, context)},
        ]
        completion = self.openai.chat.completions.parse(
            model=self.model,
            temperature=config.TEMPERATURE,
            messages=messages,
            response_format=LLMAnswer,
            # Sticky routing: keep calls on the provider holding the warm cache.
            extra_body={"session_id": config.CAG_SESSION_ID},
        )
        _log_cache_usage(getattr(completion, "usage", None))

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            # A refusal or an unparseable response is a no-answer, not a crash.
            return LLMAnswer(
                answer=config.NO_ANSWER_TEXT, has_sufficient_evidence=False, citations=[]
            )
        return parsed

    # -- validation --
    def _reconcile(
        self,
        question: str,
        completion: LLMAnswer,
        chunks: list,
        key_to_document: dict[str, str],
        core_count: int = 0,
    ) -> AskGMPResponse:
        """
        Everything the model said, checked against what was actually retrieved.

        Note the order: inline brackets are cleaned first, because those are
        what the user reads. The structured citation list is checked second and
        contributes any keys the prose did not mention, so a source used but
        cited only structurally still shows as cited in the table.
        """
        answer, used_keys, rejected_inline = validate_inline_citations(
            completion.answer or "", key_to_document, chunks
        )

        structured, rejected_structured = validate_structured_citations(
            completion.citations, key_to_document
        )
        used_keys |= {citation.document_id for citation in structured}

        rejected = list(dict.fromkeys(rejected_inline + rejected_structured))

        sources = build_sources(
            _core_cited_chunks(chunks[:core_count], key_to_document, used_keys,
                               cited_page_spans(answer, structured))
            + chunks[core_count:],
            key_to_document, cited_keys=used_keys,
        )

        # The model judged its own evidence insufficient, or validation stripped
        # the answer down to nothing. Either way, do not present prose that is
        # no longer supported.
        if not completion.has_sufficient_evidence or not answer.strip():
            return AskGMPResponse(
                answer=_no_answer_with_closest(sources),
                sources=sources,
                has_sufficient_evidence=False,
                question=question,
                rejected_citations=rejected,
            )

        return AskGMPResponse(
            answer=answer,
            sources=sources,
            has_sufficient_evidence=True,
            question=question,
            rejected_citations=rejected,
        )


def _core_cited_chunks(core_chunks: list, key_to_document: dict[str, str],
                       used_keys: set[str],
                       cited_spans: dict[str, list[tuple[int, int]]]) -> list:
    """
    The core chunks that belong in the source table.

    Every core document is in every prompt, so listing them all would bury the
    rows that matter: only cited documents show. And a cited one shows only the
    chunks on the pages it was cited for — otherwise its row would read
    "pp. 1-9" for an answer drawn from page 6.
    """
    document_to_key = {value: key for key, value in key_to_document.items()}
    by_document: dict[str, list] = {}
    for chunk in core_chunks:
        key = document_to_key.get(chunk.document_id)
        if key in used_keys:
            by_document.setdefault(key, []).append(chunk)

    shown = []
    for key, document_chunks in by_document.items():
        spans = cited_spans.get(key)
        narrowed = [
            chunk for chunk in document_chunks
            if spans and chunk.page is not None and any(
                chunk.page <= end and (chunk.page_end or chunk.page) >= start
                for start, end in spans
            )
        ]
        # No page cited (or the document has none): the whole document stands.
        shown += narrowed or document_chunks
    return shown


def _log_cache_usage(usage) -> None:
    """Cache reads and writes, so a broken prefix shows up as zero reads."""
    global last_usage
    details = getattr(usage, "prompt_tokens_details", None)
    if usage is None or details is None:
        return
    last_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "cached_tokens": getattr(details, "cached_tokens", None),
        "cache_write_tokens": getattr(details, "cache_write_tokens", None),
        "cost": getattr(usage, "cost", None),   # OpenRouter's billed USD
    }
    logger.info("LLM usage: %s", last_usage)


# The most recent call's token usage, real numbers from the provider, for
# /api/health: the size of the cached core is whatever was last read from cache.
last_usage: dict | None = None


def _invert(document_to_key: dict[str, str]) -> dict[str, str]:
    """document_id -> key becomes key -> document_id, which is how lookups run."""
    return {key: document_id for document_id, key in document_to_key.items()}


def _no_answer_with_closest(sources: list) -> str:
    """The refusal, plus a pointer at whatever did come back."""
    if not sources:
        return config.NO_ANSWER_TEXT

    lines = [config.NO_ANSWER_TEXT, "", "The closest documents I found are:"]
    for source in sources[:5]:
        pages = f", {source.pages}" if source.pages and source.pages != "N/A" else ""
        authority = f" — {source.authority}" if source.authority else ""
        lines.append(f"- {source.document_title}{authority} [{source.citation_key}{pages}]")
    return "\n".join(lines)
