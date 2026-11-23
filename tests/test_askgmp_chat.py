"""
Tests for AskGMP retrieval, answering, citation validation and the API.

No OpenAI or Qdrant calls. The service takes both clients by injection, so the
whole flow — retrieval filtering, diversification, prompt context, citation
validation, source aggregation — runs against stubs.

The tests that matter most are the ones about citations. AskGMP's one hard
promise is that it never shows a claim it cannot trace to a document and a
page, so "a hallucinated document ID is rejected" and "a page the retrieval
never returned is corrected" are the acceptance criteria, not edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


from app import config
from app.chat.citations import (
    build_sources,
    validate_inline_citations,
    validate_structured_citations,
)
from app.chat.formatting import (
    assign_citation_keys,
    build_description,
    clean_section,
    format_page_ranges,
    format_pages,
    format_pages_inline,
    merge_page_ranges,
    official_url,
)
from app.chat.schemas import AskGMPResponse, LLMAnswer, LLMCitation
from app.chat.service import AskGMPService
from app.retrieval.retriever import RetrievedChunk, diversify


# ── Fixtures ──────────────────────────────────────────────────────────────────
def chunk(
    document_id="ICH_ICH_Q9_R1",
    title="ICH Q9(R1) Quality Risk Management",
    authority="ICH",
    reference_code="ICH Q9(R1)",
    page=12,
    page_end=12,
    score=0.9,
    text="Quality risk management is a systematic process.",
    status="effective",
    **overrides,
):
    payload = {
        "document_id": document_id,
        "document_key": document_id,
        "document_title": title,
        "authority": authority,
        "jurisdiction": "International",
        "reference_code": reference_code,
        "document_type": "guideline",
        "content_role": "primary",
        "regulatory_weight": "official_guideline",
        "status": status,
        "topic": "quality risk management",
        "page": page,
        "page_end": page_end,
        "section": f"Risk Assessment (p. {page})" if page else None,
        "document_url": None,
        "source_page_url": None,
        "citation_url": None,
        "source_file": "ICH_Q9.pdf",
        "text": text,
    }
    payload.update(overrides)
    return RetrievedChunk(text=text, score=score, payload=payload)


class StubRetriever:
    """Returns a fixed chunk list and records how it was called."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, question, effective_only=None, authority=None):
        self.calls.append({"question": question, "effective_only": effective_only,
                           "authority": authority})
        return self.chunks


class StubOpenAI:
    """Minimal stand-in for client.chat.completions.parse()."""

    def __init__(self, parsed: LLMAnswer | None):
        self._parsed = parsed
        self.requests = []
        outer = self

        class _Completions:
            def parse(self, **kwargs):
                outer.requests.append(kwargs)
                message = type("M", (), {"parsed": outer._parsed})()
                choice = type("C", (), {"message": message})()
                return type("R", (), {"choices": [choice]})()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def service_for(chunks, parsed: LLMAnswer | None) -> AskGMPService:
    return AskGMPService(retriever=StubRetriever(chunks), openai_client=StubOpenAI(parsed))


# ── 1. Every successful answer has sources ────────────────────────────────────
def test_successful_answer_always_has_sources():
    parsed = LLMAnswer(
        answer="Risk management is systematic [ICH_Q9_R1, p. 12].",
        has_sufficient_evidence=True,
        citations=[LLMCitation(document_id="ICH_Q9_R1", page_start=12)],
    )
    response = service_for([chunk()], parsed).answer("What is QRM?")

    assert isinstance(response, AskGMPResponse)
    assert response.has_sufficient_evidence
    assert response.sources, "a substantive answer must never be source-less"
    assert response.sources[0].document_id == "ICH_ICH_Q9_R1"
    assert response.sources[0].cited is True


def test_answer_retains_its_inline_citation():
    parsed = LLMAnswer(
        answer="Risk management is systematic [ICH_Q9_R1, p. 12].",
        has_sufficient_evidence=True,
        citations=[],
    )
    response = service_for([chunk()], parsed).answer("What is QRM?")
    assert "[ICH_Q9_R1, p. 12]" in response.answer


# ── 2. Cited document IDs must exist in the retrieved chunks ──────────────────
def test_citation_keys_map_to_retrieved_documents():
    chunks = [chunk(), chunk(document_id="ICH_ICH_Q10", reference_code="ICH Q10", page=8)]
    keys = assign_citation_keys(chunks)

    assert keys["ICH_ICH_Q9_R1"] == "ICH_Q9_R1"
    assert keys["ICH_ICH_Q10"] == "ICH_Q10"


def test_documents_sharing_a_reference_code_get_distinct_keys():
    """ICH Q7 the guideline and ICH Q7 the Q&A must not collapse into one key."""
    chunks = [
        chunk(document_id="ICH_ICH_Q7", reference_code="ICH Q7", content_role="primary"),
        chunk(document_id="ICH_ICH_Q7_Q_A", reference_code="ICH Q7", content_role="q&a"),
    ]
    keys = assign_citation_keys(chunks)

    assert keys["ICH_ICH_Q7"] == "ICH_Q7"
    assert keys["ICH_ICH_Q7_Q_A"] == "ICH_Q7_QA"
    assert len(set(keys.values())) == 2


def test_long_document_ids_are_shortened_for_citation():
    long_id = "EUDRALEX_01_GUIDELINES_ON_GOOD_MANUFACTURING_PRACTICE_SPECIFIC_TO_ADVANCED_THERAPY"
    keys = assign_citation_keys([chunk(document_id=long_id, reference_code=None)])
    assert len(keys[long_id]) <= 44


# ── 3. Page numbers come from retrieval metadata ──────────────────────────────
def test_pages_in_sources_come_from_chunk_payloads():
    response = service_for(
        [chunk(page=12, page_end=13)],
        LLMAnswer(answer="Text [ICH_Q9_R1, pp. 12-13].", has_sufficient_evidence=True),
    ).answer("q")

    source = response.sources[0]
    assert source.page_start == 12 and source.page_end == 13
    assert source.pages == "12-13"


def test_a_cited_page_outside_retrieval_is_corrected():
    """
    The model wrote p. 99 for a document retrieved on p. 12. The document is
    right and the locator is wrong, so the page is corrected rather than the
    whole citation discarded.
    """
    parsed = LLMAnswer(
        answer="Risk management is systematic [ICH_Q9_R1, p. 99].",
        has_sufficient_evidence=True,
    )
    response = service_for([chunk(page=12, page_end=12)], parsed).answer("q")

    assert "p. 99" not in response.answer
    assert "[ICH_Q9_R1, p. 12]" in response.answer


def test_a_page_is_dropped_when_the_document_has_none():
    """JSON Q&A collections have no pages; a page cited for one is removed."""
    parsed = LLMAnswer(
        answer="The Q&A says so [EMA_GMP_GDP_QA, p. 4].", has_sufficient_evidence=True
    )
    chunks = [chunk(document_id="EMA_EMA_GMP_GDP_QA", reference_code=None,
                    page=None, page_end=None, section=None)]
    # Key derives from the document_id when there is no reference code.
    key = assign_citation_keys(chunks)["EMA_EMA_GMP_GDP_QA"]
    parsed.answer = f"The Q&A says so [{key}, p. 4]."

    response = service_for(chunks, parsed).answer("q")

    assert f"[{key}]" in response.answer
    assert "p. 4" not in response.answer
    assert response.sources[0].pages == "N/A"


# ── 4. Hallucinated citations are rejected ────────────────────────────────────
def test_unknown_document_id_is_stripped_from_the_answer():
    parsed = LLMAnswer(
        answer="Real claim [ICH_Q9_R1, p. 12]. Invented claim [ISO_9001, p. 3].",
        has_sufficient_evidence=True,
        citations=[LLMCitation(document_id="ISO_9001", page_start=3)],
    )
    response = service_for([chunk()], parsed).answer("q")

    assert "ISO_9001" not in response.answer
    assert "[ICH_Q9_R1, p. 12]" in response.answer
    assert "ISO_9001" in response.rejected_citations


def test_structured_citations_are_validated_too():
    keys = {"ICH_Q9_R1": "ICH_ICH_Q9_R1"}
    kept, rejected = validate_structured_citations(
        [LLMCitation(document_id="ICH_Q9_R1"), LLMCitation(document_id="MADE_UP")], keys
    )
    assert [c.document_id for c in kept] == ["ICH_Q9_R1"]
    assert rejected == ["MADE_UP"]


def test_a_grouped_bracket_keeps_the_valid_half():
    answer, used, rejected = validate_inline_citations(
        "Claim [ICH_Q9_R1, p. 12; FAKE_DOC, p. 2].",
        {"ICH_Q9_R1": "ICH_ICH_Q9_R1"},
        [chunk()],
    )
    assert "[ICH_Q9_R1, p. 12]" in answer
    assert "FAKE_DOC" not in answer
    assert rejected == ["FAKE_DOC"]
    assert used == {"ICH_Q9_R1"}


def test_ordinary_square_brackets_are_left_alone():
    answer, _, rejected = validate_inline_citations(
        "The operator [or their deputy] signs the record.", {}, []
    )
    assert "[or their deputy]" in answer
    assert rejected == []


# ── 5. Source table deduplicates repeated chunks ──────────────────────────────
def test_five_chunks_from_one_document_make_one_row():
    chunks = [chunk(page=p, page_end=p, score=0.9 - p / 100) for p in (11, 12, 13, 14, 15)]
    sources = build_sources(chunks, {"ICH_Q9_R1": "ICH_ICH_Q9_R1"})

    assert len(sources) == 1
    assert sources[0].pages == "11-15"


def test_two_documents_make_two_rows():
    chunks = [chunk(), chunk(document_id="ICH_ICH_Q10", reference_code="ICH Q10")]
    keys = assign_citation_keys(chunks)
    sources = build_sources(chunks, {v: k for k, v in keys.items()})
    assert len(sources) == 2


# ── 6. Contiguous pages aggregate; non-contiguous stay separate ───────────────
def test_contiguous_pages_merge():
    assert merge_page_ranges([(11, 11), (12, 12), (13, 13)]) == [(11, 13)]
    assert format_page_ranges([(11, 13)]) == "11-13"


def test_non_contiguous_pages_stay_separate():
    """4 and 40 are two different places to look; "4-40" would be a lie."""
    spans = merge_page_ranges([(4, 4), (40, 41)])
    assert spans == [(4, 4), (40, 41)]
    assert format_page_ranges(spans) == "4, 40-41"


def test_overlapping_spans_merge():
    assert merge_page_ranges([(5, 8), (7, 10)]) == [(5, 10)]


def test_source_row_shows_split_ranges():
    chunks = [chunk(page=4, page_end=4), chunk(page=40, page_end=41, score=0.5)]
    sources = build_sources(chunks, {"ICH_Q9_R1": "ICH_ICH_Q9_R1"})
    assert sources[0].pages == "4, 40-41"
    assert sources[0].page_ranges == ["4", "40-41"]


@pytest.mark.parametrize("start,end,table,inline", [
    (12, None, "12", "p. 12"),
    (12, 12, "12", "p. 12"),
    (12, 14, "12-14", "pp. 12-14"),
    (None, None, "N/A", ""),
])
def test_page_formatting(start, end, table, inline):
    assert format_pages(start, end) == table
    assert format_pages_inline(start, end) == inline


# ── 7. document_url falls back to source_page_url ─────────────────────────────
def test_official_url_prefers_document_url():
    assert official_url({
        "document_url": "https://example.org/doc.pdf",
        "source_page_url": "https://example.org/page",
    }) == "https://example.org/doc.pdf"


def test_official_url_falls_back_to_source_page_url():
    assert official_url({
        "document_url": None, "source_page_url": "https://example.org/page",
    }) == "https://example.org/page"


def test_official_url_is_none_when_registry_has_neither():
    assert official_url({"document_url": None, "source_page_url": None}) is None


def test_source_row_carries_the_resolved_url():
    chunks = [chunk(document_url=None, source_page_url="https://example.org/page")]
    source = build_sources(chunks, {"ICH_Q9_R1": "ICH_ICH_Q9_R1"})[0]
    assert source.official_url == "https://example.org/page"


# ── 8. Non-effective documents are filtered by default ────────────────────────
def test_effective_only_is_the_default_and_reaches_the_retriever():
    retriever = StubRetriever([chunk()])
    AskGMPService(
        retriever=retriever,
        openai_client=StubOpenAI(LLMAnswer(answer="x [ICH_Q9_R1, p. 12]",
                                           has_sufficient_evidence=True)),
    ).answer("q")
    assert retriever.calls[0]["effective_only"] is None  # service default -> config
    assert config.EFFECTIVE_ONLY is True


def test_retriever_builds_a_status_filter_by_default():
    """The Qdrant filter must actually carry status=effective."""
    from app.retrieval.retriever import Retriever

    captured = {}

    class FakeQdrant:
        def query_points(self, **kwargs):
            captured.update(kwargs)
            return type("R", (), {"points": []})()

    class FakeOpenAI:
        class embeddings:
            @staticmethod
            def create(model, input):
                item = type("E", (), {"embedding": [0.0] * 8})()
                return type("R", (), {"data": [item]})()

    Retriever(qdrant_client=FakeQdrant(), openai_client=FakeOpenAI()).search("data integrity")

    conditions = captured["prefetch"][0].filter.must
    statuses = [c.match.value for c in conditions if c.key == "status"]
    assert statuses == ["effective"]


def test_effective_filter_can_be_turned_off():
    from app.retrieval.retriever import Retriever

    captured = {}

    class FakeQdrant:
        def query_points(self, **kwargs):
            captured.update(kwargs)
            return type("R", (), {"points": []})()

    class FakeOpenAI:
        class embeddings:
            @staticmethod
            def create(model, input):
                item = type("E", (), {"embedding": [0.0] * 8})()
                return type("R", (), {"data": [item]})()

    Retriever(qdrant_client=FakeQdrant(), openai_client=FakeOpenAI()).search(
        "q", effective_only=False
    )
    conditions = captured["prefetch"][0].filter.must
    assert not [c for c in conditions if c.key == "status"]


# ── 9. Weak retrieval produces a no-answer rather than a guess ────────────────
def test_no_retrieval_returns_the_no_answer_without_calling_the_model():
    stub = StubOpenAI(None)
    response = AskGMPService(retriever=StubRetriever([]), openai_client=stub).answer("q")

    assert response.has_sufficient_evidence is False
    assert response.answer == config.NO_ANSWER_TEXT
    assert response.sources == []
    assert stub.requests == [], "the model must not be called with no context"


def test_model_declaring_insufficient_evidence_still_returns_sources():
    parsed = LLMAnswer(
        answer="Something loosely related.", has_sufficient_evidence=False, citations=[]
    )
    response = service_for([chunk()], parsed).answer("Unrelated question")

    assert response.has_sufficient_evidence is False
    assert config.NO_ANSWER_TEXT in response.answer
    assert "closest documents" in response.answer.lower()
    assert response.sources, "the closest matches are still worth showing"


def test_an_answer_emptied_by_validation_becomes_a_no_answer():
    """Every citation was invented, so the prose it supported cannot stand."""
    parsed = LLMAnswer(answer="[MADE_UP, p. 3]", has_sufficient_evidence=True,
                       citations=[LLMCitation(document_id="MADE_UP")])
    response = service_for([chunk()], parsed).answer("q")

    assert response.has_sufficient_evidence is False
    assert "MADE_UP" in response.rejected_citations


def test_empty_question_short_circuits():
    stub = StubOpenAI(None)
    response = AskGMPService(retriever=StubRetriever([chunk()]), openai_client=stub).answer("   ")
    assert response.has_sufficient_evidence is False
    assert stub.requests == []


# ── 10. API response matches the schema ───────────────────────────────────────
def test_api_chat_returns_the_expected_shape(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import main

    parsed = LLMAnswer(
        answer="Risk management is systematic [ICH_Q9_R1, p. 12].",
        has_sufficient_evidence=True,
        citations=[LLMCitation(document_id="ICH_Q9_R1", page_start=12)],
    )
    monkeypatch.setattr(main, "get_service", lambda: service_for([chunk()], parsed))

    body = TestClient(main.app).post("/api/chat", json={"question": "What is QRM?"}).json()

    assert set(body) >= {"answer", "sources", "has_sufficient_evidence"}
    assert body["answer"]
    assert len(body["sources"]) == 1

    source = body["sources"][0]
    for field in ("document_id", "document_title", "description", "authority",
                  "reference_code", "page_start", "page_end", "section",
                  "document_url", "source_page_url", "relevance_score"):
        assert field in source, f"{field} missing from the API source object"


def test_api_rejects_an_empty_question():
    from fastapi.testclient import TestClient

    from app.api import main

    assert TestClient(main.app).post("/api/chat", json={"question": "  "}).status_code == 422


def test_api_surfaces_backend_failure_rather_than_an_empty_answer(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import main

    class Boom:
        def answer(self, *args, **kwargs):
            raise RuntimeError("qdrant down")

    monkeypatch.setattr(main, "get_service", lambda: Boom())
    response = TestClient(main.app).post("/api/chat", json={"question": "q"})

    assert response.status_code == 502
    assert "qdrant down" in response.json()["detail"]


# ── Retrieval shaping ─────────────────────────────────────────────────────────
def test_diversify_caps_chunks_per_document():
    chunks = [chunk(page=p, score=0.9) for p in range(1, 7)]          # one document
    chunks += [chunk(document_id="WHO_TRS", reference_code="TRS 1044 Annex 2", score=0.5)]

    selected = diversify(chunks, max_chunks=4, max_per_document=2)

    assert len(selected) == 4
    assert sum(1 for c in selected if c.document_id == "ICH_ICH_Q9_R1") <= 3


def test_diversify_backfills_when_only_one_document_matches():
    """A question with a single relevant document still gets a full window."""
    chunks = [chunk(page=p, score=0.9) for p in range(1, 7)]
    selected = diversify(chunks, max_chunks=5, max_per_document=2)
    assert len(selected) == 5


def test_diversify_prefers_a_second_document_over_a_third_chunk():
    chunks = [
        chunk(page=1, score=0.99),
        chunk(page=2, score=0.98),
        chunk(page=3, score=0.97),
        chunk(document_id="WHO_TRS", reference_code="TRS 1044 Annex 2", score=0.10),
    ]
    selected = diversify(chunks, max_chunks=3, max_per_document=2)
    assert {c.document_id for c in selected} == {"ICH_ICH_Q9_R1", "WHO_TRS"}


# ── Descriptions and sections ─────────────────────────────────────────────────
def test_description_is_composed_from_metadata():
    description = build_description({
        "authority": "FDA",
        "regulatory_weight": "supporting_guidance",
        "document_type": "guidance",
        "document_title": "Process Validation: General Principles and Practices",
    })
    assert description.startswith("FDA supporting guidance")
    assert "Process Validation" in description
    assert description.endswith(".")


def test_description_appends_a_reference_code_not_already_in_the_title():
    description = build_description({
        "authority": "ICH", "document_title": "Quality Risk Management",
        "reference_code": "ICH Q9(R1)", "regulatory_weight": "official_guideline",
    })
    assert "(ICH Q9(R1))" in description


def test_section_label_drops_its_duplicated_page_suffix():
    assert clean_section("4. BUILDINGS AND FACILITIES (p. 7)") == "4. BUILDINGS AND FACILITIES"
    assert clean_section("Risk Assessment (pp. 12-14)") == "Risk Assessment"
    assert clean_section(None) is None


# ── Prompt context ────────────────────────────────────────────────────────────
def test_context_block_shows_the_citation_key_as_document_id():
    from app.chat.prompts import build_context

    chunks = [chunk()]
    keys = assign_citation_keys(chunks)
    context = build_context(chunks, keys)

    assert "[SOURCE 1]" in context
    assert "Document ID: ICH_Q9_R1" in context
    assert "Page: 12" in context
    assert "TEXT:" in context


def test_context_block_omits_fields_the_payload_lacks():
    """A "Pages: None" line would invite the model to cite page None."""
    from app.chat.prompts import build_context

    chunks = [chunk(page=None, page_end=None, reference_code=None, section=None)]
    context = build_context(chunks, assign_citation_keys(chunks))

    assert "Page:" not in context
    assert "Reference:" not in context


def test_context_truncates_an_oversized_chunk():
    from app.chat.prompts import build_context

    chunks = [chunk(text="x" * (config.MAX_CHUNK_CHARS + 500))]
    context = build_context(chunks, assign_citation_keys(chunks))
    assert len(context) < config.MAX_CHUNK_CHARS + 1000


# ── Section-style citations (documents with no pages) ─────────────────────────
def test_a_grounded_section_citation_is_kept():
    """CFR JSON has no pages, so the model cites by section instead."""
    cfr = chunk(
        document_id="FDA_21_CFR_PART_211",
        reference_code="21 CFR Part 211",
        page=None, page_end=None,
        section="211.100 Written procedures; deviations.",
        text="There shall be written procedures for production and process control.",
    )
    answer, used, rejected = validate_inline_citations(
        "Deviations must be recorded [21_CFR_PART_211, Section 211.100].",
        {"21_CFR_PART_211": "FDA_21_CFR_PART_211"},
        [cfr],
    )
    assert "[21_CFR_PART_211, Section 211.100]" in answer
    assert used == {"21_CFR_PART_211"}
    assert rejected == []


def test_an_invented_section_is_dropped_but_the_document_kept():
    cfr = chunk(
        document_id="FDA_21_CFR_PART_211", reference_code="21 CFR Part 211",
        page=None, page_end=None, section="211.100 Written procedures; deviations.",
        text="There shall be written procedures.",
    )
    answer, _, rejected = validate_inline_citations(
        "Claim [21_CFR_PART_211, Section 999.999].",
        {"21_CFR_PART_211": "FDA_21_CFR_PART_211"},
        [cfr],
    )
    assert "999.999" not in answer
    assert "[21_CFR_PART_211]" in answer
    assert rejected == []


def test_an_unknown_key_with_a_section_tail_is_still_rejected():
    """The hole this closed: a section label must not smuggle a bad key past."""
    answer, _, rejected = validate_inline_citations(
        "Claim [ISO_9001, Section 8.5].", {"ICH_Q9_R1": "ICH_ICH_Q9_R1"}, [chunk()]
    )
    assert "ISO_9001" not in answer
    assert rejected == ["ISO_9001"]


def test_prose_in_brackets_is_still_left_alone():
    for text in ("The operator [or their deputy] signs.",
                 "Record it [see local procedure] promptly."):
        answer, _, rejected = validate_inline_citations(text, {}, [])
        assert answer == text
        assert rejected == []


# ── Frontend: the source table the user actually sees ─────────────────────────
FAKE_ANSWER = {
    "answer": "Deviations must be investigated [ICH_Q9_R1, p. 12].",
    "has_sufficient_evidence": True,
    "rejected_citations": [],
    "sources": [{
        "citation_key": "ICH_Q9_R1",
        "document_id": "ICH_ICH_Q9_R1",
        "document_title": "ICH Q9(R1) Quality Risk Management",
        "description": "ICH official guideline covering quality risk management.",
        "authority": "ICH",
        "reference_code": "ICH Q9(R1)",
        "pages": "12-14",
        "section": "Risk Assessment",
        "official_url": "https://database.ich.org/q9.pdf",
        "cited": True,
        "relevance_score": 0.91,
    }],
}


@pytest.fixture
def stub_backend(monkeypatch):
    """Replace the HTTP calls so the UI renders without a running backend."""
    import requests

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(FAKE_ANSWER))
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: Response({"qdrant": "connected", "points": 1844}),
    )


def launch():
    """The app on first load — the landing state."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(BASE_DIR / "app" / "frontend" / "streamlit_app.py"),
                            default_timeout=60)
    app.run()
    return app


def submit_landing(app, question: str):
    """Type into the landing text area and press its submit button."""
    app.text_area[0].set_value(question)
    [b for b in app.button if b.label == "Ask AskGMP"][0].click().run()
    return app


def test_frontend_renders_answer_and_source_table(stub_backend):
    app = submit_landing(launch(), "Where are deviation requirements?")
    assert not app.exception

    rendered = " ".join(block.value for block in app.markdown)
    assert "[ICH_Q9_R1, p. 12]" in rendered

    assert len(app.dataframe) == 1, "one source table under the answer"
    frame = app.dataframe[0].value
    for column in ("Document ID", "Title", "Description", "Authority", "Reference",
                   "Related Page(s)", "Section", "Official Document"):
        assert column in frame.columns
    assert frame.iloc[0]["Related Page(s)"] == "12-14"
    assert frame.iloc[0]["Official Document"] == "https://database.ich.org/q9.pdf"


def test_frontend_shows_a_table_even_when_it_cannot_answer(monkeypatch):
    """The no-answer path must still show the closest documents."""
    import requests

    payload = dict(FAKE_ANSWER,
                   answer=config.NO_ANSWER_TEXT, has_sufficient_evidence=False)

    class Response:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(payload))
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: Response({"qdrant": "connected", "points": 1844}))

    app = submit_landing(launch(), "Boiling point of ethanol?")

    assert not app.exception
    assert any(config.NO_ANSWER_TEXT in w.value for w in app.warning)
    assert len(app.dataframe) == 1


# ── Two-stage chat UX ─────────────────────────────────────────────────────────
def test_landing_state_shows_the_form_and_no_chat_furniture(stub_backend):
    """
    First load is a landing page, not an empty chatbot.

    The empty transcript and the pinned input are what made this screen read as
    a generic assistant waiting to be fed, so both must be absent.
    """
    app = launch()

    assert not app.exception
    assert len(app.text_area) == 1, "the prominent first-question box"
    assert app.text_area[0].placeholder.startswith("Ask about a GMP requirement")
    assert [b.label for b in app.button if b.label == "Ask AskGMP"]

    assert len(app.chat_input) == 0, "no bottom chat bar before the first question"
    assert len(app.chat_message) == 0, "no empty transcript"
    assert len(app.dataframe) == 0

    # Suggestion chips are a button group in Streamlit's element tree.
    assert len(app.get("button_group")) == 1


def test_first_submission_switches_to_conversation_mode(stub_backend):
    app = submit_landing(launch(), "Where are deviation requirements?")

    assert not app.exception
    assert len(app.text_area) == 0, "the landing box is gone"
    assert len(app.chat_input) == 1, "the follow-up bar has taken over"
    assert app.chat_input[0].placeholder == "Ask a follow-up question…"
    assert len(app.chat_message) == 2, "the question and its answer"
    assert len(app.dataframe) == 1


def test_the_two_inputs_are_never_shown_together(stub_backend):
    """
    The transition must not flash both. Queuing the question and answering it
    on the next run is what guarantees that, so it is asserted rather than
    assumed.
    """
    app = launch()
    assert (len(app.text_area), len(app.chat_input)) == (1, 0)

    submit_landing(app, "Where are deviation requirements?")
    assert (len(app.text_area), len(app.chat_input)) == (0, 1)


def test_a_suggestion_chip_submits_immediately(stub_backend):
    app = launch()
    label = ":green[:material/verified_user:] Data integrity"

    app.get("button_group")[0].set_value(label).run()

    assert not app.exception
    assert len(app.chat_message) == 2, "the chip asked its question outright"
    assert len(app.text_area) == 0
    rendered = " ".join(block.value for block in app.markdown)
    assert "Which documents discuss data integrity?" in rendered


def test_follow_up_uses_the_chat_input_and_appends(stub_backend):
    app = submit_landing(launch(), "First question?")
    app.chat_input[0].set_value("And the FDA view?").run()

    assert not app.exception
    assert len(app.chat_message) == 4, "both turns kept, neither duplicated"
    assert len(app.dataframe) == 2


def test_both_inputs_send_the_same_request(monkeypatch):
    """One submission pipeline: the landing form and the chat bar must agree."""
    import requests

    sent = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return FAKE_ANSWER

    def post(url, json=None, **kwargs):
        sent.append(json)
        return Response()

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: type(
        "R", (), {"raise_for_status": lambda s: None,
                  "json": lambda s: {"qdrant": "connected", "points": 1844}})())

    app = submit_landing(launch(), "From the landing form")
    app.chat_input[0].set_value("From the chat bar").run()

    assert [item["question"] for item in sent] == ["From the landing form",
                                                   "From the chat bar"]
    # Same payload shape from both paths, filters included.
    assert {k for k in sent[0]} == {k for k in sent[1]}
    assert sent[0]["effective_only"] is True


def test_clear_conversation_returns_to_the_landing_state(stub_backend):
    app = submit_landing(launch(), "Where are deviation requirements?")
    assert len(app.chat_message) == 2

    [b for b in app.sidebar.button if b.label == "Clear conversation"][0].click().run()

    assert not app.exception
    assert len(app.chat_message) == 0
    assert len(app.dataframe) == 0
    assert len(app.chat_input) == 0, "the bottom bar is hidden again"
    assert len(app.text_area) == 1, "the landing box is back"
    assert app.text_area[0].value == "", "and empty"


def test_clearing_does_not_re_fire_the_last_suggestion(stub_backend):
    """
    A pill keeps its selection in widget state. Left set, redrawing the landing
    page would immediately re-ask the same question.
    """
    app = launch()
    app.get("button_group")[0].set_value(":orange[:material/science:] FDA on OOS").run()
    assert len(app.chat_message) == 2

    [b for b in app.sidebar.button if b.label == "Clear conversation"][0].click().run()

    assert len(app.chat_message) == 0, "clearing did not re-submit the chip"
    assert app.get("button_group")[0].value is None


def test_empty_landing_submission_is_refused(stub_backend):
    app = launch()
    [b for b in app.button if b.label == "Ask AskGMP"][0].click().run()

    assert not app.exception
    assert len(app.chat_message) == 0, "whitespace is not a question"
    assert len(app.text_area) == 1, "still on the landing page"
    assert any("Type a question" in w.value for w in app.warning)


def test_sources_to_frame_has_the_specified_columns():
    # Imported from app.frontend.tables, not from the Streamlit script:
    # importing the script executes it, which leaves Streamlit's global state
    # part-way through a form and breaks the next test's widgets.
    from app.frontend.tables import TABLE_COLUMNS, sources_to_frame

    frame = sources_to_frame(FAKE_ANSWER["sources"])
    assert list(frame.columns)[: len(TABLE_COLUMNS)] == TABLE_COLUMNS
    assert frame.iloc[0]["Document ID"] == "ICH_Q9_R1"


def test_sources_to_frame_survives_missing_fields():
    from app.frontend.tables import sources_to_frame

    frame = sources_to_frame([{"document_id": "X", "document_title": "Y"}])
    assert frame.iloc[0]["Related Page(s)"] == "N/A"
    assert frame.iloc[0]["Document ID"] == "X"


# ── Landing-page document uploader ────────────────────────────────────────────
def test_landing_shows_the_uploader_under_the_question_box(stub_backend):
    app = launch()
    uploaders = app.get("file_uploader")

    assert len(uploaders) == 1
    assert uploaders[0].label == "Upload documents"
    assert len(app.sidebar.toggle) == 2, "the OCR toggle sits beside the filters"


def test_records_to_frame_shows_where_each_record_came_from():
    from app.frontend.tables import RECORD_COLUMNS, records_to_frame

    frame = records_to_frame([
        {"text": "Storage temperature 2-8 C",
         "meta": {"locator": "Sheet 'Specs', row 2", "kind": "row"}},
        {"text": "OOS RESULT CONFIRMED",
         "meta": {"locator": "Figure 1 on slide 3", "kind": "image", "ocr": True}},
    ])

    assert list(frame.columns) == RECORD_COLUMNS
    assert frame.iloc[0]["Where"] == "Sheet 'Specs', row 2"
    assert bool(frame.iloc[1]["OCR"])      # pandas hands back np.True_
    assert not bool(frame.iloc[0]["OCR"])


def test_records_to_frame_limits_the_preview():
    from app.frontend.tables import records_to_frame

    many = [{"text": f"row {n}", "meta": {"kind": "row"}} for n in range(50)]
    assert len(records_to_frame(many, limit=12)) == 12
