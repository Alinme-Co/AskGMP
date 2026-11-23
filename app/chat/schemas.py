"""
schemas.py — Typed contracts for AskGMP.

Two families of model, and the split between them is the point.

`LLMAnswer` is what the language model is allowed to produce: prose, a
confidence flag, and citations expressed as nothing more than a document key
and a page range. `AskGMPSource` is what the API returns, and every field on it
except the model's page choice is copied from the Qdrant payload.

The model is never asked for a title, an authority, a reference code or a URL.
Not because it would usually get them wrong, but because there is no way to
tell, from the output alone, the difference between a URL it read and a URL it
composed — and a plausible-looking citation to a regulation is worse than an
admission of ignorance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── What the model is allowed to say ──────────────────────────────────────────
class LLMCitation(BaseModel):
    """One citation as the model expresses it: a key and, at most, pages."""

    document_id: str = Field(
        description="The exact 'Document ID' value from one of the supplied "
                    "sources. Never invent one."
    )
    page_start: int | None = Field(
        default=None,
        description="First page supporting the claim, copied from the source "
                    "block. Null when the source shows no page.",
    )
    page_end: int | None = Field(
        default=None,
        description="Last page, when the claim spans a range. Null otherwise.",
    )
    section: str | None = Field(
        default=None, description="Section heading from the source block, if shown."
    )


class LLMAnswer(BaseModel):
    """The model's structured response."""

    answer: str = Field(
        description="The answer, with inline citations in [DOCUMENT_ID, p. N] form."
    )
    has_sufficient_evidence: bool = Field(
        description="False when the supplied sources do not actually support an "
                    "answer to the question."
    )
    citations: list[LLMCitation] = Field(
        default_factory=list,
        description="Every source used, once per distinct page range.",
    )


# ── What AskGMP returns ───────────────────────────────────────────────────────
class AskGMPSource(BaseModel):
    """
    One row of the source table.

    `citation_key` is the short handle used inside the answer text;
    `document_id` is the registry's full identifier. They are kept separate
    because the registry's ids run to a hundred characters
    ("EUDRALEX_01_GUIDELINES_ON_GOOD_MANUFACTURING_PRACTICE_SPECIFIC_TO_...")
    and are unusable mid-sentence.
    """

    citation_key: str
    document_id: str
    document_title: str
    description: str | None = None
    # "upload" or "knowledge_base". A site's own procedure and a published
    # regulation must never be presented as the same kind of authority.
    origin: str = "knowledge_base"
    authority: str | None = None
    jurisdiction: str | None = None
    document_type: str | None = None
    reference_code: str | None = None
    topic: str | None = None
    status: str | None = None
    regulatory_weight: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    pages: str | None = None            # display form: "12", "12-14", "N/A"
    page_ranges: list[str] = Field(default_factory=list)
    section: str | None = None
    document_url: str | None = None
    source_page_url: str | None = None
    official_url: str | None = None     # document_url, else source_page_url
    source_file: str | None = None
    relevance_score: float | None = None
    cited: bool = False                 # did the answer actually reference it?


class AskGMPResponse(BaseModel):
    answer: str
    sources: list[AskGMPSource] = Field(default_factory=list)
    has_sufficient_evidence: bool = True
    question: str | None = None
    rejected_citations: list[str] = Field(
        default_factory=list,
        description="Document keys the model cited that were not in the "
                    "retrieved set. Removed from the answer text; surfaced here "
                    "so the failure is visible rather than silent.",
    )


# ── API ───────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    effective_only: bool | None = Field(
        default=None,
        description="Restrict to status='effective'. Defaults to the server "
                    "setting, which is on.",
    )
    authority: str | None = Field(
        default=None, description="Optional authority filter, e.g. 'FDA'."
    )
    document_hashes: list[str] = Field(
        default_factory=list,
        description="SHA-256 hashes of uploaded documents to search alongside "
                    "the knowledge base. The document itself stays on the "
                    "server; only its hash travels.",
    )


# ── Document summaries ────────────────────────────────────────────────────────
class SummaryPoint(BaseModel):
    """One thing the document says, and where it says it."""

    point: str = Field(description="A single substantive statement, one sentence.")
    locator: str | None = Field(
        default=None,
        description="Where it appears — copied verbatim from the source block's "
                    "Where line. Null if the block showed none.",
    )


class LLMDocumentSummary(BaseModel):
    """What the model is allowed to say about an uploaded document."""

    title: str = Field(description="The document's own title, or a short "
                                   "descriptive one if it states none.")
    document_kind: str = Field(
        description="What kind of document this is in the author's own terms — "
                    "SOP, batch record, deviation report, specification, "
                    "protocol, guidance. 'Unclear' if it does not say."
    )
    summary: str = Field(description="What the document covers, in 2-4 sentences.")
    key_points: list[SummaryPoint] = Field(
        default_factory=list, description="Up to 8 substantive points."
    )


class DocumentSummary(BaseModel):
    """A summary as AskGMP returns it."""

    file_hash: str
    filename: str
    title: str
    document_kind: str
    summary: str
    key_points: list[SummaryPoint] = Field(default_factory=list)
    chunk_count: int = 0
    truncated: bool = Field(
        default=False,
        description="True when the document was longer than the summariser's "
                    "budget and only its opening was read.",
    )
    error: str | None = None


class SummaryRequest(BaseModel):
    file_hash: str = Field(min_length=8)
    filename: str | None = None
