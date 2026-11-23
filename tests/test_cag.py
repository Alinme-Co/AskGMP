"""
CAG-RAG: the cached core library in front of retrieval.

What has to hold for caching to work at all is that the core prefix is
byte-identical between questions; what has to hold for AskGMP's promise is that
core citations validate like any other and do not flood the source table.
"""

from __future__ import annotations

from test_askgmp_chat import StubOpenAI, chunk

from app.chat.schemas import LLMAnswer
from app.chat.service import AskGMPService


class StubCore:
    def __init__(self, chunks):
        self._chunks = chunks
        self.document_ids = list(dict.fromkeys(c.document_id for c in chunks))

    def chunks(self):
        return self._chunks


class RecordingRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append(kwargs)
        return self.chunks


CORE = [
    chunk(document_id="EUDRALEX_EU_GMP_CHAPTER_4", title="EU GMP Chapter 4 Documentation",
          authority="EMA", reference_code="EU GMP Chapter 4", page=3, page_end=3,
          text="Batch records should be retained for at least one year after expiry."),
    chunk(document_id="EUDRALEX_EU_GMP_CHAPTER_4", title="EU GMP Chapter 4 Documentation",
          authority="EMA", reference_code="EU GMP Chapter 4", page=7, page_end=7,
          text="Packaging records should be kept for each batch."),
    chunk(document_id="FDA_21_CFR_PART_211", title="21 CFR Part 211", authority="FDA",
          reference_code="21 CFR Part 211", page=None, page_end=None,
          text="211.180 Records shall be retained."),
]


def _service(retrieved, parsed):
    openai = StubOpenAI(parsed)
    retriever = RecordingRetriever(retrieved)
    return AskGMPService(retriever=retriever, openai_client=openai,
                         core=StubCore(CORE)), openai, retriever


def _answer(text, ok=True):
    return LLMAnswer(answer=text, has_sufficient_evidence=ok, citations=[])


def test_core_prefix_is_identical_across_questions_and_only_it_is_cached():
    service, openai, _ = _service([chunk()], _answer("x"))
    service.answer("Batch record retention?")
    service.retriever.chunks = [chunk(document_id="WHO_TRS_992_ANNEX_4", reference_code="")]
    service.answer("Hold time studies?")

    first, second = (r["messages"][0]["content"] for r in openai.requests)
    assert first == second
    assert "cache_control" not in first[0]
    assert first[1]["cache_control"] == {"type": "ephemeral"}
    assert "EU_GMP_CHAPTER_4" in first[1]["text"]
    assert all(r["extra_body"]["session_id"] for r in openai.requests)


def test_retrieved_core_hits_are_dropped_not_sent_twice():
    core_hit = chunk(document_id="FDA_21_CFR_PART_211", reference_code="21 CFR Part 211",
                     text="RETRIEVED COPY of 211.67")
    service, openai, _ = _service([core_hit, chunk()], _answer("x"))
    service.answer("q")
    user = openai.requests[0]["messages"][1]["content"]
    assert "RETRIEVED COPY" not in user              # core text appears once, cached
    assert "Document ID: ICH_Q9_R1" in user          # long-tail hit still sent


def test_authority_filter_falls_back_to_plain_rag():
    core_hit = chunk(document_id="FDA_21_CFR_PART_211", reference_code="21 CFR Part 211",
                     text="RETRIEVED COPY of 211.67")
    service, openai, _ = _service([core_hit], _answer("x"))
    service.answer("q", authority="FDA")
    assert len(openai.requests[0]["messages"][0]["content"]) == 1    # no core block
    assert "RETRIEVED COPY" in openai.requests[0]["messages"][1]["content"]


def test_core_citation_validates_and_uncited_core_is_not_a_source():
    parsed = _answer("Keep batch records a year past expiry [EU_GMP_CHAPTER_4, p. 3].")
    service, _, _ = _service([chunk()], parsed)
    response = service.answer("Batch record retention?")

    assert "[EU_GMP_CHAPTER_4, p. 3]" in response.answer
    ids = [s.document_id for s in response.sources]
    assert "EUDRALEX_EU_GMP_CHAPTER_4" in ids          # cited core: shown
    assert "FDA_21_CFR_PART_211" not in ids            # uncited core: hidden
    assert "ICH_ICH_Q9_R1" in ids                      # retrieved: always shown

    row = next(s for s in response.sources if s.document_id == "EUDRALEX_EU_GMP_CHAPTER_4")
    assert row.pages == "3"                            # cited page, not the whole doc


def test_core_cited_without_a_page_shows_the_whole_document():
    service, _, _ = _service([chunk()], _answer("Records are kept [EU_GMP_CHAPTER_4]."))
    response = service.answer("q")
    row = next(s for s in response.sources if s.document_id == "EUDRALEX_EU_GMP_CHAPTER_4")
    assert row.pages == "3, 7"


def test_cited_page_spans_reads_prose_and_structured_citations():
    from app.chat.citations import cited_page_spans
    from app.chat.schemas import LLMCitation

    spans = cited_page_spans(
        "A [KEY_A, pp. 4-5; KEY_B]. B [or a deputy].",
        [LLMCitation(document_id="KEY_C", page_start=9)],
    )
    assert spans == {"KEY_A": [(4, 5)], "KEY_C": [(9, 9)]}


def test_colliding_retrieved_document_does_not_steal_a_core_key():
    clash = chunk(document_id="OTHER_DOC", reference_code="EU GMP Chapter 4")
    service, openai, _ = _service([clash], _answer("x"))
    service.answer("q")
    core_text = openai.requests[0]["messages"][0]["content"][1]["text"]
    assert "Document ID: EU_GMP_CHAPTER_4\n" in core_text
    assert "Document ID: OTHER_DOC" in openai.requests[0]["messages"][1]["content"]


def test_eval_score_needs_an_expected_cited_document_and_every_fact():
    from scripts.eval_cag import CASES, score

    parsed = _answer("Keep batch records one year after expiry [EU_GMP_CHAPTER_4, p. 3].")
    service, _, _ = _service([chunk()], parsed)
    response = service.answer("q")

    result = score(response, ["EUDRALEX_EU_GMP_CHAPTER_4"], ["one year"])
    assert result["hit"] and result["facts"] and result["rejected"] == 0
    assert not score(response, ["EUDRALEX_EU_GMP_CHAPTER_4"], ["one year", "five years"])["facts"]
    assert not score(response, ["FDA_21_CFR_PART_211"], [])["hit"]   # uncited core
    assert all(expected and facts for _, expected, facts in CASES)
