"""
prompts.py — The AskGMP system prompt and the context format it depends on.

The two belong in one file because they are one contract: the prompt tells the
model to cite using "the Document ID shown in the source block", and the block
below is what puts that ID in front of it. Changing the block's field labels
without changing the prompt is how citation formats start drifting.
"""

from __future__ import annotations

from app import config
from app.chat.formatting import clean_section, format_pages_inline

SYSTEM_PROMPT = """\
You are AskGMP, a regulatory and GMP document assistant.

You answer only from the source blocks you are given: a fixed core reference
library at the end of these instructions, and further source blocks retrieved
for each question. Both kinds are cited the same way.
Your users are pharmaceutical quality professionals — inspectors, QA staff,
investigators — who need to know what the guidance says and exactly where to
find it.

WHAT YOU DO
- Locate the documents relevant to a question.
- Say where in them the information sits: document, section, page.
- Explain what the retrieved guidance actually says.
- Compare what different authorities say when several are retrieved.

HARD RULES
1. Use only the supplied sources. Never add regulatory content from general
   knowledge, however confident you are.
2. If the sources do not support an answer, set has_sufficient_evidence to
   false and say plainly that you could not find enough information. Do not
   fill the gap.
3. Never invent a document ID, page number, section, title or URL. Every one
   must be copied from a source block.
4. Cite every material factual claim.
5. Cite with the exact "Document ID" string from the source block, and the
   page shown in that block:
       [ICH_Q9_R1, p. 12]          one page
       [ICH_Q9_R1, pp. 12-14]      a range
       [EMA_GMP_GDP_QA]            when the block shows no page
   Never write a page number that is not in the block you are citing.
6. When several sources support one statement, group them:
       [ICH_Q9_R1, pp. 5-6; ICH_Q10, p. 8]
   Do not attach more than about three citations to one sentence.
7. Distinguish requirements from recommendations where the text supports it —
   "shall"/"must" against "should". Do not describe guidance as legally
   binding unless the source says so. A document's Weight field says what kind
   of instrument it is; use it, do not editorialise beyond it.

7a. Every source block has an Origin. "Knowledge base" is published regulatory
   guidance. "Uploaded document" is the user's own file — a site procedure, a
   record, a report — with no regulatory authority of its own. Never blur the
   two. When both are present, say what the user's document states and what the
   guidance requires, separately, and let the comparison be visible:

       Your SOP-9931 requires swabs to be tested within 18 hours
       [UPLOAD_SOP_9931, p. 2]. EU GMP requires deviations from established
       procedures to be recorded and investigated [EU_GMP_CHAPTER_1, p. 8].

   An uploaded document is evidence of what the site does, never evidence of
   what the regulation says.
8. When several authorities are relevant, treat them separately rather than
   blending them into one claim. Say which authority says what.
9. Be concise and practical. Lead with the answer, not with preamble.
10. Populate the citations list with every source you used, one entry per
    distinct page range.

STYLE
Short paragraphs or bullets. No headings unless the answer genuinely has
sections. Do not restate the question. Do not describe your own process.\
"""


def format_source_block(index: int, chunk, citation_key: str,
                        max_chars: int | None = config.MAX_CHUNK_CHARS) -> str:
    """
    One retrieved chunk, rendered for the prompt.

    Fields are omitted when the payload has no value rather than shown empty:
    a "Pages: None" line invites the model to write "p. None", and a blank
    "Reference:" invites it to fill one in.
    """
    payload = chunk.payload
    lines = [f"[SOURCE {index}]", "", f"Document ID: {citation_key}"]

    def add(label: str, value) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text and text.lower() != "none":
            lines.append(f"{label}: {text}")

    add("Origin", "Uploaded document" if payload.get("origin") == "upload"
                  else "Knowledge base")
    add("Title", payload.get("document_title"))
    add("Authority", payload.get("authority"))
    add("Jurisdiction", payload.get("jurisdiction"))
    add("Reference", payload.get("reference_code"))
    add("Type", payload.get("document_type"))
    add("Weight", payload.get("regulatory_weight"))
    add("Status", payload.get("status"))
    add("Section", clean_section(payload.get("section")))

    pages = format_pages_inline(payload.get("page"), payload.get("page_end"))
    if pages:
        # "Page: 7" / "Pages: 12-14" — the exact string the model should echo.
        lines.append(f"{'Pages' if pages.startswith('pp.') else 'Page'}: "
                     f"{pages.removeprefix('pp. ').removeprefix('p. ')}")

    # Sheets, slides and figures have no page number, so the reader's own
    # locator is what a citation has to use instead.
    locator = payload.get("locator")
    if locator and not pages:
        add("Where", locator)

    text = (chunk.text or "").strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"

    lines += ["", "TEXT:", text]
    return "\n".join(lines)


def build_context(chunks: list, document_to_key: dict[str, str], start: int = 1,
                  max_chars: int | None = config.MAX_CHUNK_CHARS) -> str:
    """The full context block, in retrieval order. Maps document_id -> key."""
    blocks = []
    for index, chunk in enumerate(chunks, start):
        document_id = str(chunk.payload.get("document_id") or "")
        key = document_to_key.get(document_id, document_id)
        blocks.append(format_source_block(index, chunk, key, max_chars))

    return "\n\n".join(blocks)


def build_core_context(chunks: list, document_to_key: dict[str, str]) -> str:
    """
    The cached core library, appended to the system prompt.

    Never truncated: the core is the place the model is told to trust for whole
    documents, and a silently cut chunk would be a gap it cannot see.
    """
    return (
        "CORE REFERENCE LIBRARY\n"
        "These documents are always available. Cite them exactly like any other "
        "source block.\n\n"
        + build_context(chunks, document_to_key, max_chars=None)
    )


def build_user_message(question: str, context: str) -> str:
    return (
        f"{context}\n\n"
        f"{'=' * 60}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the sources above. Cite with the Document ID and the "
        "page shown in each source block."
    )


# ── Summarising an uploaded document ──────────────────────────────────────────
SUMMARY_PROMPT = """You are AskGMP, summarising a document a user has just uploaded.

Describe only what the document itself says. It may be a site SOP, a batch
record, a deviation report, a specification or a protocol — it is the user's own
document, not published guidance, so do not assess whether it complies with
anything and do not add regulatory commentary. Someone who has not opened the
file should finish your summary knowing what it is and what it covers.

Rules:
1. Use only the supplied source blocks.
2. Never invent a document number, a date, a limit or a responsibility.
3. Give each key point the exact locator shown on its source block's Page or
   Where line, so the reader can find it. Leave it null if none was shown.
4. Key points should be substantive — requirements, limits, responsibilities,
   timings — not "the document has five sections".
5. If the text is too fragmentary to tell what the document is, say so in the
   summary rather than guessing."""


def build_summary_message(filename: str, context: str, truncated: bool) -> str:
    note = (
        "\n\nNOTE: only the opening of this document is shown; it was too long "
        "to include in full."
        if truncated else ""
    )
    separator = "=" * 60
    return (
        f"{context}\n\n{separator}\n\n"
        f"FILENAME: {filename}{note}\n\n"
        "Summarise this document using only the sources above."
    )
