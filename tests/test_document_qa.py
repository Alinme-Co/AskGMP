"""
Tests for RAG over uploaded documents: chunking, in-memory search, merged
answering, and summarisation.

No OpenAI or Qdrant. The embedding client is stubbed with a deterministic
bag-of-words vectoriser, which is enough to make cosine similarity behave
sensibly — a chunk sharing words with the question ranks above one that does
not — without a network call.

The behaviour that matters most is the separation between a user's own document
and published guidance. An uploaded SOP is evidence of what a site does; a
regulation is evidence of what is required. Several tests below exist purely to
keep those from being presented as the same kind of thing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


from app import config
from app.chat.schemas import (
    AskGMPResponse,
    LLMAnswer,
    LLMCitation,
    LLMDocumentSummary,
    SummaryPoint,
)
from app.chat.service import AskGMPService
from app.core.constants import ORIGIN_UPLOAD
from app.documents.models import KIND_PARAGRAPH, record
from app.uploads.search import (
    UploadIndex,
    document_id_for,
)
from app.uploads.storage import original_stem
from app.uploads.summarise import DocumentSummariser
from app.retrieval.retriever import RetrievedChunk


# ── Stubs ─────────────────────────────────────────────────────────────────────
_WORD = re.compile(r"[a-z0-9]+")
_VOCAB = [
    "swab", "sample", "zyntarol", "hold", "hours", "temperature", "deviation",
    "recorded", "investigated", "quality", "cleaning", "validation", "form",
    "batch", "release", "training", "calibration", "equipment", "supplier",
]


def _vector(text: str) -> list[float]:
    """A deterministic bag-of-words vector — enough for cosine to be meaningful."""
    words = set(_WORD.findall(text.lower()))
    return [1.0 if term in words else 0.0 for term in _VOCAB] + [0.1]


class StubEmbeddings:
    def create(self, model, input):
        items = [type("E", (), {"embedding": _vector(text)})() for text in input]
        return type("R", (), {"data": items})()


class StubEmbeddingClient:
    embeddings = StubEmbeddings()


class StubChatClient:
    """Returns a fixed parsed object from chat.completions.parse()."""

    def __init__(self, parsed):
        self.parsed = parsed
        self.requests = []
        outer = self

        class _Completions:
            def parse(self, **kwargs):
                outer.requests.append(kwargs)
                message = type("M", (), {"parsed": outer.parsed})()
                return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls = []

    def retrieve(self, question, effective_only=None, authority=None):
        self.calls.append(question)
        return list(self.chunks)


def kb_chunk(document_id="ICH_ICH_Q9_R1", text="Deviations shall be investigated.",
             page=8, score=0.9, **overrides):
    payload = {
        "document_id": document_id,
        "document_title": "ICH Q9 Quality Risk Management",
        "authority": "ICH",
        "reference_code": "ICH Q9",
        "document_type": "guideline",
        "regulatory_weight": "official_guideline",
        "status": "effective",
        "content_role": "primary",
        "page": page, "page_end": page,
        "section": "Risk Assessment",
        "document_url": None, "source_page_url": None,
        "source_file": "ICH_Q9.pdf", "text": text,
    }
    payload.update(overrides)
    return RetrievedChunk(text=text, score=score, payload=payload)


SOP_RECORDS = [
    record("SOP-9931 Cleaning Validation", kind="heading", locator="SOP-9931"),
    record(
        "Swab samples for ZYNTAROL-40 shall be held at 4 degrees C and tested "
        "within 18 hours of collection.",
        kind=KIND_PARAGRAPH, section="4. Sampling", locator="4. Sampling",
    ),
    record(
        "Any departure from the hold time shall be recorded on form QA-22 and "
        "reported to the Quality Unit within one working day.",
        kind=KIND_PARAGRAPH, section="5. Departures", locator="5. Departures",
    ),
    record(
        "Equipment calibration records are retained for five years by the "
        "engineering department.",
        kind=KIND_PARAGRAPH, section="9. Calibration", locator="9. Calibration",
    ),
]

HASH = "a" * 64


@pytest.fixture
def index():
    upload = UploadIndex(openai_client=StubEmbeddingClient())
    upload.add_records(SOP_RECORDS, "SOP_9931.docx", HASH)
    return upload


# ── Identity ──────────────────────────────────────────────────────────────────
def test_upload_ids_are_prefixed_so_the_citation_says_where_it_came_from():
    key = document_id_for("SOP_9931.docx", "f29405befb4f00")
    assert key.startswith("UPLOAD_")
    assert "SOP_9931" in key


def test_the_storage_hash_is_not_repeated_in_the_id():
    """
    Files are stored as "<stem>-<hash12>.docx". Reading one back off disk gives
    only that name, so without stripping it the id became
    UPLOAD_SOP_9931_F29405BEFB4F_F29405.
    """
    assert original_stem("SOP_9931-f29405befb4f.docx") == "SOP_9931"
    assert document_id_for("SOP_9931-f29405befb4f.docx", "f29405befb4f00") == \
           document_id_for("SOP_9931.docx", "f29405befb4f00")


def test_two_documents_get_different_ids():
    assert document_id_for("a.docx", "1" * 64) != document_id_for("a.docx", "2" * 64)


# ── Chunking and indexing ─────────────────────────────────────────────────────
def test_an_upload_is_chunked_and_embedded_once(index):
    document = index.get(HASH)
    assert document.chunk_count > 0
    assert len(document.vectors) == document.chunk_count
    assert len(document.sparse) == document.chunk_count


def test_re_adding_the_same_hash_does_not_re_embed():
    calls = []

    class Counting(StubEmbeddingClient):
        class embeddings:
            @staticmethod
            def create(model, input):
                calls.append(len(input))
                return StubEmbeddings().create(model, input)

    upload = UploadIndex(openai_client=Counting())
    upload.add_records(SOP_RECORDS, "SOP.docx", HASH)
    upload.add_records(SOP_RECORDS, "SOP.docx", HASH)
    assert len(calls) == 1, "embedding a document twice costs money for nothing"


def test_chunks_keep_their_provenance(index):
    chunks = index.all_chunks(HASH)
    sections = [chunk.payload.get("section") for chunk in chunks]
    assert any(section and "Sampling" in section for section in sections)


# ── Search ────────────────────────────────────────────────────────────────────
def test_search_finds_the_relevant_passage(index):
    results = index.search("What is the swab sample hold time?", [HASH])

    assert results
    assert "18 hours" in results[0].text


def test_search_ranks_an_unrelated_section_lower(index):
    results = index.search("swab sample hold hours", [HASH])
    texts = [chunk.text for chunk in results]

    hold = next(i for i, t in enumerate(texts) if "18 hours" in t)
    calibration = next((i for i, t in enumerate(texts) if "calibration" in t.lower()), 99)
    assert hold < calibration


def test_search_marks_results_as_uploads(index):
    result = index.search("swab hold time", [HASH])[0]

    assert result.payload["origin"] == ORIGIN_UPLOAD
    assert result.payload["document_id"].startswith("UPLOAD_")


def test_uploads_carry_no_invented_regulatory_metadata(index):
    """A site's draft SOP must never look like published guidance."""
    payload = index.search("swab hold time", [HASH])[0].payload

    for field in ("authority", "reference_code", "regulatory_weight",
                  "document_url", "source_page_url", "status"):
        assert payload[field] is None, f"{field} must stay empty for an upload"
    assert payload["official_source"] is False


def test_search_respects_the_chunk_budget(index):
    assert len(index.search("swab", [HASH], limit=1)) <= 1


def test_search_returns_nothing_without_a_question_or_document(index):
    assert index.search("", [HASH]) == []
    assert index.search("swab", []) == []


def test_an_unknown_hash_is_ignored_rather_than_raising(index):
    assert index.search("swab", ["deadbeef" * 8]) == []


# ── Answering with both origins ───────────────────────────────────────────────
def service_with(upload_index, kb_chunks, parsed):
    return AskGMPService(
        retriever=StubRetriever(kb_chunks),
        openai_client=StubChatClient(parsed),
        upload_index=upload_index,
    )


def test_the_answer_can_cite_both_the_upload_and_the_regulation(index):
    upload_key = None
    for chunk in index.all_chunks(HASH):
        upload_key = chunk.payload["document_id"]
        break

    parsed = LLMAnswer(
        answer=f"Your SOP requires 18 hours [{upload_key}]. "
               "GMP requires deviations to be investigated [ICH_Q9].",
        has_sufficient_evidence=True,
        citations=[],
    )
    response = service_with(index, [kb_chunk()], parsed).answer(
        "Is a 26-hour hold a deviation?", document_hashes=[HASH]
    )

    assert isinstance(response, AskGMPResponse)
    origins = {source.origin for source in response.sources}
    assert origins == {"upload", "knowledge_base"}
    assert upload_key in response.answer
    assert "ICH_Q9" in response.answer


def test_upload_sources_are_labelled_in_the_table(index):
    parsed = LLMAnswer(answer="Something.", has_sufficient_evidence=True)
    response = service_with(index, [kb_chunk()], parsed).answer(
        "swab hold time", document_hashes=[HASH]
    )

    uploaded = [s for s in response.sources if s.origin == "upload"]
    assert uploaded
    assert uploaded[0].authority is None
    assert uploaded[0].document_title == "SOP 9931"


def test_the_uploaded_context_is_put_before_the_corpus(index):
    """
    The user's own document leads, because a question like "is this a
    deviation?" is about their document first.
    """
    parsed = LLMAnswer(answer="x", has_sufficient_evidence=True)
    service = service_with(index, [kb_chunk()], parsed)
    service.answer("swab hold time", document_hashes=[HASH])

    context = service._openai.requests[0]["messages"][1]["content"]
    assert context.index("Uploaded document") < context.index("Knowledge base")


def test_the_context_labels_each_source_with_its_origin(index):
    parsed = LLMAnswer(answer="x", has_sufficient_evidence=True)
    service = service_with(index, [kb_chunk()], parsed)
    service.answer("swab hold time", document_hashes=[HASH])

    context = service._openai.requests[0]["messages"][1]["content"]
    assert "Origin: Uploaded document" in context
    assert "Origin: Knowledge base" in context


def test_a_question_without_uploads_is_unchanged(index):
    parsed = LLMAnswer(answer="x [ICH_Q9, p. 8]", has_sufficient_evidence=True)
    response = service_with(index, [kb_chunk()], parsed).answer("What is QRM?")

    assert all(source.origin == "knowledge_base" for source in response.sources)


def test_the_answer_still_works_when_only_an_upload_matches(index):
    """No corpus hit is not the same as no evidence."""
    parsed = LLMAnswer(answer="Your SOP says 18 hours.", has_sufficient_evidence=True)
    response = service_with(index, [], parsed).answer(
        "swab hold time", document_hashes=[HASH]
    )

    assert response.has_sufficient_evidence
    assert response.sources and response.sources[0].origin == "upload"


def test_a_broken_upload_index_does_not_lose_the_corpus_answer():
    """A failure reading an upload must not cost the knowledge-base answer."""
    class Exploding:
        def search(self, *args, **kwargs):
            raise RuntimeError("disk gone")

    parsed = LLMAnswer(answer="x [ICH_Q9, p. 8]", has_sufficient_evidence=True)
    response = AskGMPService(
        retriever=StubRetriever([kb_chunk()]),
        openai_client=StubChatClient(parsed),
        upload_index=Exploding(),
    ).answer("q", document_hashes=[HASH])

    assert response.has_sufficient_evidence
    assert response.sources


def test_an_invented_upload_citation_is_still_rejected(index):
    parsed = LLMAnswer(
        answer="Claim [UPLOAD_SOMETHING_ELSE, p. 3].", has_sufficient_evidence=True
    )
    response = service_with(index, [kb_chunk()], parsed).answer(
        "swab", document_hashes=[HASH]
    )

    assert "UPLOAD_SOMETHING_ELSE" not in response.answer
    assert "UPLOAD_SOMETHING_ELSE" in response.rejected_citations


# ── Summarising ───────────────────────────────────────────────────────────────
def summariser_with(index, parsed):
    return DocumentSummariser(upload_index=index, openai_client=StubChatClient(parsed))


def test_a_summary_is_produced_without_a_question(index):
    parsed = LLMDocumentSummary(
        title="SOP-9931 Cleaning Validation",
        document_kind="SOP",
        summary="Covers swab sampling and departures.",
        key_points=[SummaryPoint(point="Swabs tested within 18 hours.",
                                 locator="4. Sampling")],
    )
    summary = summariser_with(index, parsed).summarise(HASH, "SOP_9931.docx")

    assert summary.error is None
    assert summary.document_kind == "SOP"
    assert summary.chunk_count > 0
    assert summary.key_points[0].locator == "4. Sampling"


def test_a_summary_locator_the_document_lacks_is_dropped(index):
    """Same rule as citations: a locator is a promise the reader can follow."""
    parsed = LLMDocumentSummary(
        title="T", document_kind="SOP", summary="s",
        key_points=[SummaryPoint(point="Real point.", locator="Appendix Z, p. 400")],
    )
    summary = summariser_with(index, parsed).summarise(HASH)

    assert summary.key_points[0].point == "Real point."
    assert summary.key_points[0].locator is None


def test_summarising_an_unknown_document_reports_rather_than_raises(index):
    summary = summariser_with(index, None).summarise("f" * 64, "ghost.docx")

    assert summary.error
    assert summary.chunk_count == 0


def test_a_long_document_is_truncated_and_says_so(monkeypatch):
    monkeypatch.setattr(config, "MAX_SUMMARY_CHUNKS", 2)

    upload = UploadIndex(openai_client=StubEmbeddingClient())
    many = [
        record(f"Section {n}. " + ("Cleaning validation requirement text. " * 60),
               kind=KIND_PARAGRAPH, section=f"Section {n}")
        for n in range(8)
    ]
    upload.add_records(many, "long.docx", "b" * 64)

    parsed = LLMDocumentSummary(title="T", document_kind="SOP", summary="s")
    summary = summariser_with(upload, parsed).summarise("b" * 64)

    assert summary.truncated
    assert summary.chunk_count > 2


def test_the_summariser_is_told_not_to_judge_compliance(index):
    from app.chat.prompts import SUMMARY_PROMPT

    assert "not published guidance" in SUMMARY_PROMPT
    assert "do not assess whether it complies" in SUMMARY_PROMPT


# ── The prompt contract ───────────────────────────────────────────────────────
def test_the_system_prompt_separates_uploads_from_guidance():
    from app.chat.prompts import SYSTEM_PROMPT

    assert "Uploaded document" in SYSTEM_PROMPT
    assert "never evidence of" in SYSTEM_PROMPT


def test_the_source_block_shows_a_locator_when_there_is_no_page(index):
    from app.chat.prompts import format_source_block

    chunk = index.search("swab hold time", [HASH])[0]
    block = format_source_block(1, chunk, "UPLOAD_X")

    assert "Origin: Uploaded document" in block
    assert "Where:" in block or "Section:" in block


# ── API contract ──────────────────────────────────────────────────────────────
def test_chat_request_accepts_document_hashes():
    from app.chat.schemas import ChatRequest

    request = ChatRequest(question="q", document_hashes=["a" * 64])
    assert request.document_hashes == ["a" * 64]
    assert ChatRequest(question="q").document_hashes == []


def test_the_api_passes_hashes_through(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import main

    seen = {}

    class Service:
        def answer(self, question, effective_only=None, authority=None,
                   document_hashes=None):
            seen["hashes"] = document_hashes
            return AskGMPResponse(answer="ok", sources=[], has_sufficient_evidence=True)

    monkeypatch.setattr(main, "get_service", lambda: Service())
    TestClient(main.app).post(
        "/api/chat", json={"question": "q", "document_hashes": ["abc123"]}
    )

    assert seen["hashes"] == ["abc123"]


def test_the_summary_endpoint_returns_the_schema(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import main
    from app.chat.schemas import DocumentSummary

    class Summariser:
        def summarise(self, file_hash, filename=None):
            return DocumentSummary(
                file_hash=file_hash, filename=filename or "f.docx",
                title="SOP-9931", document_kind="SOP",
                summary="Covers sampling.", chunk_count=3,
            )

    monkeypatch.setattr(main, "get_summariser", lambda: Summariser())
    body = TestClient(main.app).post(
        "/api/documents/summary", json={"file_hash": "a" * 64, "filename": "f.docx"}
    ).json()

    assert body["title"] == "SOP-9931"
    assert body["document_kind"] == "SOP"
    assert body["chunk_count"] == 3


# ── Frontend table ────────────────────────────────────────────────────────────
def test_the_source_table_shows_where_each_row_came_from():
    from app.frontend.tables import TABLE_COLUMNS, sources_to_frame

    frame = sources_to_frame([
        {"citation_key": "UPLOAD_SOP", "origin": "upload", "document_title": "SOP"},
        {"citation_key": "ICH_Q9", "origin": "knowledge_base", "document_title": "Q9"},
    ])

    assert TABLE_COLUMNS[0] == "Origin"
    assert frame.iloc[0]["Origin"] == "Your document"
    assert frame.iloc[1]["Origin"] == "Knowledge base"
