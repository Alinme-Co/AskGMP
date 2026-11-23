"""
streamlit_app.py — the AskGMP chat interface.

    streamlit run app/frontend/streamlit_app.py

Talks to the FastAPI backend over HTTP rather than importing the service, so
the two halves stay separable and the API contract is exercised by the app that
uses it. Set ASKGMP_API_URL if the backend is not on localhost:8000.

Each assistant turn renders as prose with inline citations, followed by the
related-documents table. The table is the deliverable as much as the prose is:
a regulatory answer that cannot be traced to a document and a page is not
usable, so sources render underneath every answer, including the ones where
AskGMP declines to answer.

Two-stage interaction
---------------------
With no conversation yet, the page is a landing form: a large text area and a
submit button, no transcript and no bottom chat bar. After the first question
it becomes an ordinary chat, with follow-ups typed into the pinned input.

Both inputs go through queue_question(); there is one submission pipeline, not
two. The stage is derived from state rather than stored as a flag —
see `has_conversation` — so it cannot disagree with the transcript.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# `streamlit run` executes this file as a script rather than importing it as
# `app.frontend.streamlit_app`, so the project root is not on the path and
# `import app.config` would fail. This is the one place in the project that
# adjusts sys.path, and it is an entry point rather than a library module.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import config  # noqa: E402
from app.documents import UPLOAD_TYPES  # noqa: E402
from app.frontend.tables import (  # noqa: E402
    records_to_frame,
    sources_to_frame,
)
from app.uploads import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    content_hash,
    process_uploads,
)

st.set_page_config(page_title="AskGMP", page_icon=":material/biotech:", layout="wide")

SUGGESTIONS = {
    ":blue[:material/policy:] Deviation investigations":
        "Where can I find requirements about deviation investigations?",
    ":green[:material/verified_user:] Data integrity":
        "Which documents discuss data integrity?",
    ":orange[:material/science:] FDA on OOS":
        "What does FDA say about out of specification investigations?",
    ":violet[:material/memory:] Computerised systems":
        "Where can I find requirements for computerised systems?",
}

SUGGESTION_KEY = "suggestion_pills"
LANDING_TEXT_KEY = "landing_question"
UPLOADER_KEY = "document_uploader"

# Records shown per document in the preview. A 300-page scan produces hundreds;
# the panel is for confirming the read worked, not for reading the document.
PREVIEW_RECORDS = 12


# ── Sources table ─────────────────────────────────────────────────────────────
def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No supporting documents were retrieved for this question.")
        return

    cited = sum(1 for source in sources if source.get("cited"))
    st.markdown(
        f"**Sources / related documents** &nbsp;·&nbsp; {len(sources)} document(s)"
        + (f", {cited} cited in the answer" if cited else "")
    )

    st.dataframe(
        sources_to_frame(sources),
        hide_index=True,
        column_config={
            "Origin": st.column_config.TextColumn(
                "Origin",
                help="Whether the passage came from your upload or from the "
                     "indexed regulatory knowledge base.",
                width="small",
            ),
            "Document ID": st.column_config.TextColumn(
                "Document ID", help="The identifier used in the answer's citations.",
                width="medium",
            ),
            "Title": st.column_config.TextColumn("Title", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Related Page(s)": st.column_config.TextColumn(
                "Related Page(s)", help="Pages the retrieved passages came from."
            ),
            "Official Document": st.column_config.LinkColumn(
                "Official Document",
                help="The publisher's own copy. Empty where the registry has no URL yet.",
                display_text="Open",
            ),
            "Cited": st.column_config.CheckboxColumn(
                "Cited", help="Referenced in the answer above."
            ),
            "Relevance": st.column_config.NumberColumn("Relevance", format="%.3f"),
        },
    )

    missing = [s for s in sources if not s.get("official_url")]
    if missing:
        st.caption(
            f":material/link_off: {len(missing)} of {len(sources)} document(s) have no "
            "official URL in the registry yet, so they can be quoted but not linked."
        )


# ── Backend ───────────────────────────────────────────────────────────────────
def ask_backend(question: str, effective_only: bool, authority: str | None) -> dict:
    """
    One question, plus whatever the user has uploaded.

    Only hashes travel. The documents were written to data/uploads/ at upload
    time and the backend finds them by hash, so a 40 MB scan is not re-posted
    with every follow-up question.
    """
    response = requests.post(
        f"{config.API_URL}/api/chat",
        json={
            "question": question,
            "effective_only": effective_only,
            "authority": authority,
            "document_hashes": document_hashes(),
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def document_hashes() -> list[str]:
    """Hashes of the uploads that were read successfully."""
    return [
        document["file_hash"]
        for document in st.session_state.documents
        if not document.get("error")
    ]


def fetch_summary(file_hash: str, filename: str) -> dict:
    """What the document says, for when no question has been asked yet."""
    response = requests.post(
        f"{config.API_URL}/api/documents/summary",
        json={"file_hash": file_hash, "filename": filename},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


# ── State ─────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "documents" not in st.session_state:
    st.session_state.documents = []


def queue_question(question: str | None) -> bool:
    """
    The single entry point for a new question.

    Both the landing form and the follow-up chat input call this, so there is
    one submission pipeline regardless of which stage the user is in.

    The question is parked in state and answered on the *next* run rather than
    immediately. That is what keeps the two stages from overlapping: the run
    that accepts a landing-form submission has already drawn the text area, so
    answering inline would leave the landing form sitting above the first
    answer, with the chat bar appearing below it at the same time.
    """
    question = (question or "").strip()
    if not question:
        return False
    st.session_state.pending_question = question
    return True


def reset_conversation() -> None:
    """Back to the landing state, with nothing left over to re-fire."""
    st.session_state.messages = []
    st.session_state.pending_question = None
    # The pill keeps its selection in widget state. Left set, it would
    # re-submit the moment the landing page redraws.
    st.session_state[SUGGESTION_KEY] = None
    st.session_state[LANDING_TEXT_KEY] = ""


# ── Rendering ─────────────────────────────────────────────────────────────────
def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))


def answer_pending(effective_only: bool, authority: str | None) -> None:
    """
    Render and record the queued turn.

    Renders inline *and* appends to the transcript, which is why no rerun
    follows: the history loop above has already drawn the earlier turns, so
    drawing this one here shows it exactly once.
    """
    question = st.session_state.pending_question

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base…"):
            try:
                payload = ask_backend(question, effective_only, authority)
            except Exception as exc:             # noqa: BLE001
                payload = {
                    "answer": f":material/error: Could not reach AskGMP: {exc}",
                    "sources": [],
                }

        answer = payload.get("answer", "")
        sources = payload.get("sources", [])

        if payload.get("has_sufficient_evidence") is False:
            st.warning(answer)
        else:
            st.markdown(answer)

        if payload.get("rejected_citations"):
            st.caption(
                ":material/warning: Removed citation(s) that did not match a "
                f"retrieved document: {', '.join(payload['rejected_citations'])}"
            )

        render_sources(sources)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.session_state.pending_question = None


# ── Document upload ───────────────────────────────────────────────────────────
def ingest_uploads(files, use_ocr: bool) -> int:
    """
    Read whatever is new and remember it. Returns how many were read.

    Streamlit hands the same UploadedFile objects back on every rerun, so the
    content hash decides what is actually new. Without that check a scanned PDF
    would be OCR-ed again on every widget interaction.
    """
    if not files:
        return 0

    known = {document["file_hash"] for document in st.session_state.documents}
    fresh = [item for item in files if content_hash(item.getvalue()) not in known]
    if not fresh:
        return 0

    read = process_uploads(fresh, use_ocr=use_ocr)
    for document in read:
        entry = {
            **document.to_dict(),
            "path": str(document.path),
            "records_data": document.records,
            "summary_text": None,
        }
        # Summarise straight away: someone who uploads before typing anything
        # wants to know what the system made of the file, and "6 records" does
        # not answer that.
        if not document.error:
            try:
                entry["summary_text"] = fetch_summary(
                    document.file_hash, document.filename
                )
            except Exception as exc:             # noqa: BLE001
                entry["summary_error"] = str(exc)
        st.session_state.documents.append(entry)
    return len(read)


def render_documents() -> None:
    """What was read out of each uploaded file, and where it came from."""
    documents = st.session_state.documents
    if not documents:
        return

    st.markdown(f"**Uploaded documents** &nbsp;·&nbsp; {len(documents)}")

    for index, document in enumerate(documents):
        summary = document.get("summary") or {}
        records = document.get("records_data") or []

        if document.get("error"):
            headline = f":material/error: {document['filename']} — {document['error']}"
        else:
            bits = [f"{summary.get('records', 0)} record(s)",
                    f"{summary.get('characters', 0):,} characters"]
            if summary.get("ocr_records"):
                bits.append(f"{summary['ocr_records']} OCR")
            headline = (f":material/description: {document['filename']} "
                        f"({document['format']}) — {', '.join(bits)}")

        with st.expander(headline, expanded=index == len(documents) - 1):
            if document.get("error"):
                st.error(document["error"])
                continue

            written = document.get("summary_text") or {}
            if written.get("error"):
                st.warning(f"Could not summarise: {written['error']}")
            elif written.get("summary"):
                st.markdown(
                    f"**{written.get('title') or document['filename']}**"
                    f"  &nbsp;·&nbsp; *{written.get('document_kind', 'document')}*"
                )
                st.markdown(written["summary"])
                points = written.get("key_points") or []
                if points:
                    st.markdown("**Key points**")
                    for item in points:
                        where = f"  — *{item['locator']}*" if item.get("locator") else ""
                        st.markdown(f"- {item['point']}{where}")
                if written.get("truncated"):
                    st.caption(
                        ":material/info: Only the opening of this document was "
                        "summarised; questions still search all of it."
                    )
            elif document.get("summary_error"):
                st.caption(f"Summary unavailable: {document['summary_error']}")

            kinds = summary.get("kinds") or {}
            if kinds:
                st.caption(
                    "Content found: "
                    + ", ".join(f"{count} × {kind}" for kind, count in kinds.items())
                )

            st.dataframe(
                records_to_frame(records, limit=PREVIEW_RECORDS),
                hide_index=True,
                column_config={
                    "Where": st.column_config.TextColumn(
                        "Where", help="Page, sheet and cell, slide, table or figure.",
                        width="medium",
                    ),
                    "Text": st.column_config.TextColumn("Text", width="large"),
                    "OCR": st.column_config.CheckboxColumn(
                        "OCR", help="Text recovered from a picture."
                    ),
                },
            )
            if len(records) > PREVIEW_RECORDS:
                st.caption(f"Showing {PREVIEW_RECORDS} of {len(records)} records.")

    st.caption(
        ":material/search: Your uploads are searched alongside the knowledge base. "
        "Ask a question about them and the answer will cite both, keeping your "
        "own documents and published guidance clearly apart."
    )


def render_uploader(use_ocr: bool) -> None:
    """The upload control, directly beneath the question box."""
    files = st.file_uploader(
        "Upload documents",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=UPLOADER_KEY,
        help=(
            "PDF, Word, Excel, PowerPoint, HTML, XLIFF, Markdown, JSON, CSV or "
            f"text. Up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per file. "
            "Text inside scans and pictures is read with OCR."
        ),
    )

    if files:
        with st.spinner("Reading documents…"):
            added = ingest_uploads(files, use_ocr=use_ocr)
        if added:
            st.rerun()

    render_documents()


def render_landing() -> None:
    """
    The first-question state: suggestions, then one prominent text area.

    No transcript and no bottom chat bar — an empty chat frame is what made
    this screen read as a generic chatbot waiting to be fed.
    """
    selected = st.pills(
        "Try asking:",
        list(SUGGESTIONS),
        key=SUGGESTION_KEY,
        label_visibility="collapsed",
    )
    if selected and queue_question(SUGGESTIONS[selected]):
        st.rerun()

    # border=True is Streamlit's own rounded, padded container — the app has no
    # custom CSS, and hand-rolling one here would only drift from the theme.
    with st.form("first_question", clear_on_submit=True, border=True):
        question = st.text_area(
            "Your question",
            key=LANDING_TEXT_KEY,
            height=150,
            placeholder="Ask about a GMP requirement, document, process, or topic…",
            label_visibility="collapsed",
        )
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.caption("Ctrl+Enter to submit")
            submitted = st.form_submit_button(
                "Ask AskGMP",
                type="primary",
                icon=":material/arrow_forward:",
            )

    if submitted:
        if queue_question(question):
            st.rerun()
        else:
            st.warning("Type a question first.")

    render_uploader(use_ocr=st.session_state.get("upload_ocr", True))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("AskGMP")
    st.caption("Grounded answers from the indexed GMP knowledge base.")

    effective_only = st.toggle(
        "Effective documents only",
        value=True,
        help="Excludes superseded, withdrawn and draft guidance.",
    )

    authority = st.selectbox(
        "Authority",
        ["All", "European Commission", "FDA", "EMA", "ICH", "WHO", "PIC/S", "MHRA"],
        help="Restrict retrieval to one publisher.",
    )

    st.toggle(
        "Read text inside images",
        value=True,
        key="upload_ocr",
        help="OCR scanned pages, screenshots and diagrams in uploaded "
             "documents. Slower, but a scanned SOP is unreadable without it.",
    )

    if st.session_state.documents:
        st.divider()
        # The upload panel only renders on the landing page, so once the
        # conversation starts this is the only reminder that the user's own
        # documents are still being searched with every question.
        searched = sum(1 for d in st.session_state.documents if not d.get("error"))
        st.caption(
            f"{len(st.session_state.documents)} document(s) uploaded"
            + (f" — {searched} searched with every question" if searched else "")
        )
        for document in st.session_state.documents:
            icon = ":material/error:" if document.get("error") else ":material/description:"
            records = (document.get("summary") or {}).get("records", 0)
            st.caption(f"{icon} {document['filename']} — {records} record(s)")
        if st.button("Remove uploads", icon=":material/delete:"):
            st.session_state.documents = []
            st.session_state[UPLOADER_KEY] = []
            st.rerun()

    st.divider()
    try:
        health = requests.get(f"{config.API_URL}/api/health", timeout=5).json()
        if health.get("qdrant") == "connected":
            st.success(f":material/database: {health['points']:,} chunks indexed")
        else:
            st.error(f"Qdrant {health.get('qdrant')}")
    except Exception:                            # noqa: BLE001
        st.error(f":material/cloud_off: Backend unreachable at {config.API_URL}")
        st.caption("Start it with `uvicorn app.api.main:app --port 8000`")

    if st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        # A queued question counts: the sidebar draws before it is answered, and
        # that run ends without a rerun, so checking messages alone left the
        # button disabled after the first answer.
        disabled=not (st.session_state.messages or st.session_state.pending_question),
    ):
        reset_conversation()
        st.rerun()

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("AskGMP")
st.caption(
    "Ask about GMP requirements and I will answer only from the indexed "
    "regulatory documents, with the document and page behind every claim."
)

# Derived, never stored. A queued question already counts as a conversation, so
# the run that answers the first question never draws the landing form.
has_conversation = bool(st.session_state.messages) or st.session_state.pending_question

if not has_conversation:
    render_landing()
else:
    render_history()

    if st.session_state.pending_question:
        answer_pending(effective_only, None if authority == "All" else authority)

    follow_up = st.chat_input("Ask a follow-up question…", submit_mode="disable")
    if queue_question(follow_up):
        st.rerun()
