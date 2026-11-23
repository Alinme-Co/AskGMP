"""
registry.py — KnowledgeBase/_registry/documents_registry.csv is the source of
truth for what gets indexed and for every piece of metadata about it.

The indexer this replaced inferred authority, family, document type and title
from folder names and filename regexes. That worked while the corpus was two
folders of EudraLex and ICH PDFs; with seven authorities, five file formats and
regulatory state (status, effective_date, supersedes_document_id) that no
filename can express, guessing is no longer defensible. A document that becomes
`superseded` must change the index even though not one byte of the PDF changed,
and only the registry knows that.

So: nothing here parses a filename for meaning. The registry supplies the
metadata; this module's whole job is to load it, resolve each row to a file on
disk, and decide which rows become indexable units.

    load_registry()        raw rows, normalised types, NaN -> None
    discover_documents()   -> Discovery(documents, missing, unregistered,
                                        skipped_representations, not_ingested)

Selection rules, in order:

  1. ingest AND is_preferred_source must both be true. Either flag alone
     excludes the row, which is what stops a document held in several formats
     from being indexed twice.
  2. The file KnowledgeBase/<local_path> must exist. A missing file warns and
     is skipped; it does not abort the run.
  3. Each surviving row becomes one unit, keyed by artifact_id — unique per
     file, unlike document_id, which a guideline and its Q&A may share.
     resolve_duplicates() remains as a safety net for a registry that still
     has two live rows under one document_id.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, NamedTuple

# Columns the indexer relies on. Missing ones are created empty rather than
# raising, so a registry written by an older build of build_document_registry.py
# still loads.
REGISTRY_COLUMNS = [
    "document_id",
    "artifact_id",
    "document_title",
    "authority",
    "jurisdiction",
    "source_family",
    "document_type",
    "content_role",
    "reference_code",
    "topic",
    "version",
    "publication_date",
    "effective_date",
    "status",
    "regulatory_weight",
    "language",
    "file_type",
    "local_path",
    "source_page_url",
    "document_url",
    "official_source",
    "page_count",
    "text_extracted",
    "needs_ocr",
    "supersedes_document_id",
    "is_preferred_source",
    "ingest",
    "notes",
]

# Fields that must round-trip into the Qdrant payload untouched. Everything a
# citation needs, plus everything a retrieval filter might want.
METADATA_FIELDS = [
    "document_id",
    "artifact_id",
    "document_title",
    "authority",
    "jurisdiction",
    "source_family",
    "document_type",
    "content_role",
    "reference_code",
    "topic",
    "version",
    "publication_date",
    "effective_date",
    "status",
    "regulatory_weight",
    "language",
    "file_type",
    "source_page_url",
    "document_url",
    "official_source",
    "supersedes_document_id",
    "notes",
]

# Fields that, when they change, make already-indexed chunks stale even though
# the file on disk is byte-identical. `status` is the important one.
FINGERPRINT_FIELDS = [
    "document_id",
    "document_title",
    "authority",
    "jurisdiction",
    "source_family",
    "document_type",
    "content_role",
    "reference_code",
    "topic",
    "version",
    "publication_date",
    "effective_date",
    "status",
    "regulatory_weight",
    "source_page_url",
    "document_url",
    "supersedes_document_id",
]

# File formats the extraction dispatcher understands.
SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".md", ".json", ".pptx", ".txt"}

# When one logical document exists in several formats, index the most
# structured one. A JSON Q&A file already carries the question/answer split and
# a per-record source URL; the PDF of the same content has to have all of that
# recovered by guesswork.
FORMAT_PREFERENCE = ["json", "md", "markdown", "html", "htm", "txt", "pptx", "pdf"]

TRUE_STRINGS = {"true", "t", "yes", "y", "1"}


class Discovery(NamedTuple):
    documents: list[dict]               # indexable units, registry metadata attached
    missing: list[dict]                 # registry rows whose file is not on disk
    unregistered: list[str]             # files on disk with no registry row
    skipped_representations: list[dict] # alternate formats of a chosen document
    not_ingested: list[dict]            # ingest is false
    total_rows: int


# ── Value normalisation ───────────────────────────────────────────────────────
def clean_value(value: Any) -> Any:
    """
    NaN, NaT, empty strings and the literal "nan" all become None.

    pandas hands back float('nan') for every empty CSV cell, and a NaN in a
    Qdrant payload is neither JSON-serialisable nor filterable. Normalising once
    here means nothing downstream has to remember to check.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "<na>"}:
            return None
        return text
    # numpy / pandas scalars
    if hasattr(value, "item"):
        try:
            return clean_value(value.item())
        except (ValueError, AttributeError):
            pass
    return value


def as_bool(value: Any, default: bool = False) -> bool:
    value = clean_value(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in TRUE_STRINGS


# ── Loading ───────────────────────────────────────────────────────────────────
def load_registry(registry_path: Path) -> list[dict]:
    """Read the registry CSV (or parquet) into a list of plain dicts."""
    import pandas as pd

    if not registry_path.exists():
        raise FileNotFoundError(f"Document registry not found: {registry_path}")

    if registry_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(registry_path)
    else:
        # utf-8-sig: build_document_registry.py writes a BOM, and without this
        # the first column arrives named "﻿document_id".
        frame = pd.read_csv(registry_path, encoding="utf-8-sig", dtype=object)

    for column in REGISTRY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    rows: list[dict] = []
    for position, record in enumerate(frame.to_dict(orient="records")):
        row = {key: clean_value(value) for key, value in record.items()}
        row["_row"] = position + 2                      # 1-based, past the header
        row["ingest"] = as_bool(record.get("ingest"))
        row["official_source"] = as_bool(record.get("official_source"), default=True)
        row["needs_ocr"] = as_bool(record.get("needs_ocr"))
        # Defaults to True: a registry without the column must not exclude
        # every row in it.
        row["is_preferred_source"] = as_bool(
            record.get("is_preferred_source"), default=True
        )
        rows.append(row)

    return rows


# ── Duplicate representations ─────────────────────────────────────────────────
def resolve_duplicates(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Collapse rows that share a document_id, returning (kept, skipped).

    Two genuinely different situations arrive looking identical, and telling
    them apart matters:

      * One logical document held in several formats. FDA_21_CFR_PART_210 is on
        disk as .html, .json and .md — the same regulation three times. Indexing
        all three triples the embedding bill and lets one document fill every
        retrieval slot. Keep the most structured format; report the rest.

      * Two different documents whose IDs collided, because the ID is derived
        from a reference code. ICH_ICH_Q7 is both "Good Manufacturing Practice
        for APIs" and its separate "Questions and Answers"; PICS_PE_009_17 is
        both the Introduction and the Annexes. Dropping one would delete real
        guidance from the knowledge base.

    The discriminator is the file extension. Alternate representations of one
    document are, necessarily, in different formats — so a document_id holding
    two rows of the *same* format is a collision, not a duplicate. Those rows
    are all kept and given distinct `document_key`s; the ID stays as the
    registry recorded it.

    This is now a safety net rather than the main path. The registry resolves
    both cases itself: alternate representations are marked ingest=False and
    is_preferred_source=False, and colliding documents have been given distinct
    ids (ICH_ICH_Q7 / ICH_ICH_Q7_Q_A). Nothing below fires on the current
    registry — it exists so that a hand-edited or regenerated one cannot
    silently double-index a document.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("document_id") or ""), []).append(row)

    kept: list[dict] = []
    skipped: list[dict] = []

    for document_id, group in groups.items():
        for row in group:
            row["document_key"] = _document_key(row)

        if len(group) == 1:
            kept.append(group[0])
            continue

        suffixes = [Path(str(row["local_path"])).suffix.lower().lstrip(".") for row in group]

        if len(set(suffixes)) < len(suffixes):
            # Same format twice: an ID collision between distinct documents.
            for row in group:
                if not row.get("artifact_id"):
                    stem = Path(str(row["local_path"])).stem
                    row["document_key"] = f"{document_id}::{stem}"
                row["id_collision"] = True
            kept.extend(group)
            continue

        ranked = sorted(
            group,
            key=lambda row: (
                _format_rank(Path(str(row["local_path"])).suffix),
                str(row["local_path"]),
            ),
        )
        winner, losers = ranked[0], ranked[1:]
        kept.append(winner)
        for row in losers:
            row["superseded_by_path"] = winner["local_path"]
            skipped.append(row)

    kept.sort(key=lambda row: str(row["local_path"]))
    skipped.sort(key=lambda row: str(row["local_path"]))
    return kept, skipped


def _document_key(row: dict) -> str:
    """
    The stable identity of one indexable unit.

    `artifact_id` is the registry's own per-file key and is unique across every
    row, which is exactly what state tracking and chunk IDs need. `document_id`
    is not a substitute: it identifies the *logical* document, so a guideline
    and its Q&A can legitimately share one. Falling back to document_id keeps a
    registry written before artifact_id existed working.
    """
    return str(row.get("artifact_id") or row.get("document_id") or row.get("local_path"))


def _format_rank(suffix: str) -> int:
    name = suffix.lower().lstrip(".")
    return FORMAT_PREFERENCE.index(name) if name in FORMAT_PREFERENCE else len(FORMAT_PREFERENCE)


# ── Discovery ─────────────────────────────────────────────────────────────────
def discover_documents(
    knowledge_base_dir: Path,
    registry_path: Path,
    limit: int | None = None,
    rows: list[dict] | None = None,
) -> Discovery:
    """
    Registry-driven discovery. `rows` is an injection point for tests; normally
    the registry is read from disk.
    """
    rows = list(rows) if rows is not None else load_registry(registry_path)
    total_rows = len(rows)

    # Either flag alone excludes a row. is_preferred_source marks the format
    # that represents a logical document, so a row that is not the preferred
    # source is an alternate representation even if someone left ingest=True on
    # it by mistake — which is precisely the double-indexing this guards.
    not_ingested = [
        row for row in rows
        if not row["ingest"] or not row["is_preferred_source"]
    ]
    candidates = [
        row for row in rows
        if row["ingest"] and row["is_preferred_source"]
    ]

    present: list[dict] = []
    missing: list[dict] = []
    for row in candidates:
        local_path = row.get("local_path")
        if not local_path:
            row["problem"] = "local_path is empty"
            missing.append(row)
            continue

        path = (knowledge_base_dir / str(local_path)).resolve()
        if not path.is_file():
            row["problem"] = "file not found"
            missing.append(row)
            continue

        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            row["problem"] = f"unsupported format '{path.suffix}'"
            missing.append(row)
            continue

        row["path"] = str(path)
        present.append(row)

    kept, skipped = resolve_duplicates(present)

    documents = [_to_entry(row, knowledge_base_dir) for row in kept]
    if limit:
        documents = documents[:limit]

    return Discovery(
        documents               = documents,
        missing                 = missing,
        unregistered            = find_unregistered(knowledge_base_dir, rows),
        skipped_representations = skipped,
        not_ingested            = not_ingested,
        total_rows              = total_rows,
    )


def _to_entry(row: dict, knowledge_base_dir: Path) -> dict:
    """One indexable unit: registry metadata plus what extraction needs."""
    entry = {field: row.get(field) for field in METADATA_FIELDS}
    entry.update({
        "document_key" : row.get("document_key") or row.get("document_id"),
        "path"         : row["path"],
        "local_path"   : str(row["local_path"]),
        "source_file"  : Path(str(row["local_path"])).name,
        "suffix"       : Path(str(row["local_path"])).suffix.lower(),
        "needs_ocr"    : row.get("needs_ocr", False),
        "id_collision" : row.get("id_collision", False),
    })
    return entry


def find_unregistered(knowledge_base_dir: Path, rows: Iterable[dict]) -> list[str]:
    """
    Files sitting in KnowledgeBase/ that no registry row claims.

    Reported, never indexed. The registry is the review gate: a file that
    appeared in the folder without going through build_document_registry.py has
    not been looked at by anyone, and quietly embedding it would put unreviewed
    text in front of a quality investigator.
    """
    if not knowledge_base_dir.is_dir():
        return []

    registered = {
        str(row["local_path"]).replace("\\", "/").lower()
        for row in rows
        if row.get("local_path")
    }

    unregistered: list[str] = []
    for path in sorted(knowledge_base_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(knowledge_base_dir)
        if "_registry" in relative.parts:
            continue
        if relative.as_posix().lower() not in registered:
            unregistered.append(relative.as_posix())

    return unregistered
