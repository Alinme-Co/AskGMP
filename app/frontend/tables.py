"""
tables.py — turning API objects into the frames the UI renders.

Pure functions, deliberately kept out of streamlit_app.py. Importing that
script runs it, because a Streamlit app is a script rather than a module, and
running it outside a ScriptRunner leaves Streamlit's global state part-way
through a form — which then makes an unrelated widget in the next test fail
with "st.button() can't be used in an st.form()".

So anything worth testing on its own lives here, and the app imports it.
"""

from __future__ import annotations

import pandas as pd

#: The source-table columns, in the order the brief specifies.
TABLE_COLUMNS = [
    "Origin",
    "Document ID",
    "Title",
    "Description",
    "Authority",
    "Reference",
    "Related Page(s)",
    "Section",
    "Official Document",
]

#: Columns shown for an uploaded document's extracted records.
RECORD_COLUMNS = ["Where", "Type", "OCR", "Text"]

PREVIEW_CHARS = 300


def sources_to_frame(sources: list[dict]) -> pd.DataFrame:
    """The API's source objects as the related-documents table."""
    rows = [
        {
            # Uploads carry no authority, reference or URL, so without this the
            # only visible difference between a site's own draft SOP and a
            # published regulation would be three empty cells.
            "Origin"           : ("Your document"
                                  if source.get("origin") == "upload"
                                  else "Knowledge base"),
            "Document ID"      : source.get("citation_key") or source.get("document_id"),
            "Title"            : source.get("document_title"),
            "Description"      : source.get("description"),
            "Authority"        : source.get("authority"),
            "Reference"        : source.get("reference_code"),
            "Related Page(s)"  : source.get("pages") or "N/A",
            "Section"          : source.get("section"),
            "Official Document": source.get("official_url"),
            "Cited"            : bool(source.get("cited")),
            "Relevance"        : source.get("relevance_score"),
        }
        for source in sources
    ]
    return pd.DataFrame(rows, columns=[*TABLE_COLUMNS, "Cited", "Relevance"])


def records_to_frame(records: list[dict], limit: int | None = None) -> pd.DataFrame:
    """
    Extracted records as a table, showing where each one came from.

    "Where" is the locator the reader built — "Sheet 'Specs', row 4",
    "Slide 3, table 1", "Figure 2 on page 7" — which is the whole point of
    reading provenance rather than just text.
    """
    selected = records[:limit] if limit else records
    rows = [
        {
            "Where": (item.get("meta") or {}).get("locator")
                     or item.get("section") or "—",
            "Type" : (item.get("meta") or {}).get("kind", "text"),
            "OCR"  : bool((item.get("meta") or {}).get("ocr")),
            "Text" : (item.get("text") or "")[:PREVIEW_CHARS],
        }
        for item in selected
    ]
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)
