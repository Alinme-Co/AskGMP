"""
build_document_registry.py — Scan KnowledgeBase/ and produce the document
registry that the indexer treats as its source of truth.

    python -m scripts.build_document_registry           # rebuild
    python -m scripts.build_document_registry --check   # validate only
    python -m scripts.build_document_registry --rebuild # discard curation

Two ideas hold the whole thing together.

**Derived vs curated.** Some columns are facts about the file and are recomputed
on every run: where it lives, what format it is, which authority published it,
which representation is the preferred one. Others cannot be worked out from a
filename — the official URL, the publication date, whether a guideline has been
superseded — and are for a human to fill in. Those are *curated*, and a rebuild
never overwrites a cell that already has a value.

That distinction is load-bearing. The earlier version of this script rewrote
every column from scratch, so the first `python build_document_registry.py`
after someone spent an afternoon filling in 82 document URLs would silently
erase all of them. Curated columns now survive a rebuild; `--rebuild` is the
explicit opt-out.

**document_id vs artifact_id.**

    artifact_id   one physical file. Always unique.
    document_id   one logical document. Shared by alternate representations
                  (21_CFR_Part_210 as .json, .html and .md), distinct for
                  genuinely different documents (ICH Q7 and ICH Q7 Q&A).

`is_preferred_source` picks exactly one artifact per logical document, so the
indexer never embeds the same regulation three times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

KNOWLEDGE_BASE_DIR = Path("KnowledgeBase")
OUTPUT_DIR = Path("KnowledgeBase/_registry")


DOCUMENT_COLUMNS = [
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


# Facts about the file. Recomputed on every run; anything typed into one of
# these columns is overwritten.
DERIVED_COLUMNS = {
    "document_id",
    "artifact_id",
    "authority",
    "jurisdiction",
    "source_family",
    "document_type",
    "content_role",
    "regulatory_weight",
    "language",
    "file_type",
    "local_path",
    "official_source",
    "page_count",
    "text_extracted",
    "needs_ocr",
    "is_preferred_source",
}

# Everything else is curated: the script fills a sensible starting value, and
# from then on whatever is in the cell wins. Blank a cell to have it derived
# again on the next run.
CURATED_COLUMNS = [
    column for column in DOCUMENT_COLUMNS if column not in DERIVED_COLUMNS
]


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".html",
    ".htm",
    ".json",
    ".md",
    ".pptx",
    ".txt",
}


# Used only to choose among alternate representations of THE SAME logical
# document. Higher number = preferred representation.
#
# JSON outranks everything because the structured files carry a question/answer
# or section/heading split and a per-record source URL that the PDF of the same
# content would have to have reconstructed by guesswork.
FORMAT_PRIORITY = {
    "json": 100,
    "html": 90,
    "htm": 90,
    "md": 80,
    "txt": 70,
    "pdf": 60,
    "pptx": 50,
}


SOURCE_CONFIG = {
    "eudralex_volume_4": {
        "authority": "European Commission",
        "jurisdiction": "EU",
        "source_family": "eudralex",
    },
    "ich": {
        "authority": "ICH",
        "jurisdiction": "International",
        "source_family": "ich",
    },
    "ema": {
        "authority": "EMA",
        "jurisdiction": "EU",
        "source_family": "ema",
    },
    "who": {
        "authority": "WHO",
        "jurisdiction": "International",
        "source_family": "who",
    },
    "fda": {
        "authority": "FDA",
        "jurisdiction": "US",
        "source_family": "fda",
    },
    "mhra": {
        "authority": "MHRA",
        "jurisdiction": "UK",
        "source_family": "mhra",
    },
    "pics": {
        "authority": "PIC/S",
        "jurisdiction": "International",
        "source_family": "pics",
    },
}


# ── Text helpers ─────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_id(text: str) -> str:
    """Convert arbitrary text into a stable registry identifier."""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


# Filenames that say nothing about the document. "08_here.html" is a real one:
# the EudraLex page linked the guideline behind the word "here".
JUNK_TITLES = {
    "here", "index", "download", "file", "document", "untitled",
    "default", "view", "main", "en", "pdf", "doc", "attachment",
}


def title_from_filename(path: Path) -> str:
    stem = path.stem

    # Remove leading numeric prefixes such as 01_filename / 02-filename.
    stem = re.sub(r"^\d+[_\-\s]+", "", stem)

    return normalize_text(stem)


def title_from_content(path: Path) -> str | None:
    """
    Read a real title out of the file, for when the filename has none.

    Only called when the filename is junk, so the cost is a handful of files
    rather than the whole corpus.
    """
    suffix = path.suffix.lower()

    try:
        if suffix in {".html", ".htm"}:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(
                path.read_text(encoding="utf-8", errors="replace"), "html.parser"
            )
            for candidate in (soup.title, soup.find("h1")):
                if candidate and candidate.get_text(strip=True):
                    return normalize_text(candidate.get_text(" ", strip=True))[:200]

        elif suffix == ".md":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("# "):
                    return normalize_text(line[2:])[:200]

        elif suffix == ".pdf":
            import pymupdf

            with pymupdf.open(path) as document:
                title = (document.metadata or {}).get("title") or ""
                if title.strip():
                    return normalize_text(title)[:200]
    except Exception:                            # noqa: BLE001 - best effort only
        return None

    return None


# ── Classification ───────────────────────────────────────────────────────────
def detect_source_config(relative_path: Path) -> dict:
    parts = [p.lower() for p in relative_path.parts]

    for folder_name, config in SOURCE_CONFIG.items():
        if folder_name.lower() in parts:
            return config

    return {
        "authority": None,
        "jurisdiction": None,
        "source_family": None,
    }


QA_PATTERNS = [
    "questions_and_answers",
    "questions-and-answers",
    "questions and answers",
    "_qa",
    "-qa",
    "q_and_a",
    "q-and-a",
    "q&a",
]


def detect_content_role(relative_path: Path) -> str:
    """
    What part this file plays: primary text, an annex, a Q&A companion.

    Judged from the *filename*, with the folder consulted only as a fallback.
    That split matters. The previous version tested the whole path, so every
    file under eudralex_volume_4/annexes/ picked up the folder's name and came
    out as content_role "annexes" — and because the role was appended to the
    document_id, all seventeen EU GMP annexes ended up with ids like
    EUDRALEX_09_COMPUTERISED_SYSTEMS_ANNEXES. Annex 11 is a document in its own
    right, not the annexes section of something else.
    """
    stem = relative_path.stem.lower()

    if any(pattern in stem for pattern in QA_PATTERNS):
        return "q&a"

    if "annexes" in stem:
        return "annexes"

    if "annex" in stem:
        return "annex"

    if "introduction" in stem or "_intro" in stem or "-intro" in stem:
        return "introduction"

    # Folder fallback: a single file sitting in an "annexes" folder is one
    # annex, not the whole annexes volume.
    folders = [part.lower() for part in relative_path.parts[:-1]]
    if "annexes" in folders or "annex" in folders:
        return "annex"

    return "primary"


def detect_document_type(relative_path: Path) -> str:
    text = str(relative_path).lower()
    stem = relative_path.stem.lower()

    if any(pattern in stem for pattern in QA_PATTERNS) or "cgmp_qa" in text:
        return "q&a"

    if "annex" in text:
        return "annex"

    if "21_cfr" in text or "cfr_part" in text:
        return "regulation"

    if "aide" in text or "memoire" in text:
        return "aide-memoire"

    if "guidance" in text or "guideline" in text:
        return "guidance"

    if relative_path.suffix.lower() == ".pptx":
        return "presentation"

    # JSON/Markdown are file representations, not document types. Classifying
    # them as "structured_content" would turn a storage format into document
    # semantics.
    return "guideline"


def detect_reference_code(relative_path: Path) -> str | None:
    text = relative_path.stem.upper()

    # ICH. Q8_Q9_Q10 documents cover three guidelines, so every code is kept.
    #
    # The separator class is spelled out rather than using \b: underscore is a
    # word character, so \bQ never matches in "ICH_Q7" and every ICH reference
    # code silently comes back empty.
    codes = re.findall(r"(?:^|[_\s-])Q(\d+)(?:[_\s-]?R(\d+))?", text)
    if "ICH" in text and codes:
        rendered = [
            f"Q{number}" + (f"(R{revision})" if revision else "")
            for number, revision in codes
        ]
        return "ICH " + "/".join(dict.fromkeys(rendered))

    # PIC/S PE 009
    match = re.search(r"PE[_\s-]?0*09[_\s-]?(\d+)", text)
    if match:
        return f"PE 009-{match.group(1)}"

    # PIC/S PI
    match = re.search(r"PI[_\s-]?0*(\d+)[_\s-]?(\d+)", text)
    if match:
        return f"PI {int(match.group(1)):03d}-{match.group(2)}"

    # WHO TRS
    match = re.search(r"TRS[_\s-]?(\d+).*?ANNEX[_\s-]?(\d+)", text)
    if match:
        return f"TRS {match.group(1)} Annex {match.group(2)}"

    # FDA CFR
    match = re.search(r"21[_\s-]?CFR[_\s-]?PART[_\s-]?(\d+)", text)
    if match:
        return f"21 CFR Part {match.group(1)}"

    # EU GMP chapters. The numeric filename prefix is the chapter number for
    # part_1, where chapters 1-9 are present with no gaps.
    #
    # The same inference is deliberately NOT made for the annexes folder. The
    # prefixes there are download order, not annex numbers — the veterinary
    # annexes were not downloaded, so 04_Manufacture_of_Medicinal_Gases is
    # really Annex 6. Guessing would put a wrong citation in front of the model,
    # which is worse than no citation at all.
    folders = [part.lower() for part in relative_path.parts[:-1]]
    if "part_1" in folders:
        match = re.match(r"^(\d+)[_\-\s]", relative_path.stem)
        if match and 1 <= int(match.group(1)) <= 9:
            return f"EU GMP Chapter {int(match.group(1))}"

    match = re.search(r"CH(?:APTER)?[_\s-]?0*(\d+)", text)
    if match and "GMP" in text:
        return f"EU GMP Chapter {int(match.group(1))}"

    return None


TOPIC_KEYWORDS = [
    ("data integrity", ["data_integrity", "data integrity"]),
    ("quality risk management", ["quality_risk", "qrm", "risk_management"]),
    ("validation", ["validation"]),
    ("change management", ["change_management", "change management"]),
    ("sterile manufacturing", ["sterile"]),
    ("oos", ["oos", "out_of_specification", "out of specification"]),
    ("computerised systems", ["computerised", "computerized"]),
    ("api manufacturing", ["_api", "active_substances", "active pharmaceutical"]),
    ("pharmaceutical quality system", ["pharmaceutical_quality_system", "q10"]),
    ("technology transfer", ["technology_transfer"]),
    ("water systems", ["water_for_pharmaceutical"]),
    ("hvac", ["hvac", "heating", "ventilation"]),
    ("investigational products", ["investigational"]),
    ("continuous manufacturing", ["continuous_manufacturing"]),
    ("excipients", ["excipients"]),
    ("recall / defective products", ["defective", "recall"]),
]


def detect_topic(relative_path: Path) -> str | None:
    text = str(relative_path).lower()

    for topic, keywords in TOPIC_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return topic

    return None


def detect_regulatory_weight(document_type: str, source_family: str | None) -> str:
    if document_type == "regulation":
        return "regulation"

    if document_type == "q&a":
        return "official_qa"

    if source_family in {"eudralex", "ich", "who"}:
        return "official_guideline"

    if source_family == "pics":
        return "inspection_guidance"

    return "supporting_guidance"


# ── Identity ─────────────────────────────────────────────────────────────────
def create_document_base_id(relative_path: Path, reference_code: str | None) -> str:
    """
    The logical id before any disambiguation.

    Alternate representations of one document share this; so, for the moment,
    do a guideline and its Q&A companion. assign_document_ids() separates
    those, and only those.
    """
    config = detect_source_config(relative_path)
    source = config["source_family"] or "DOC"

    if reference_code:
        base = f"{source}_{reference_code}"
    else:
        base = f"{source}_{relative_path.stem}"

    return normalize_id(base)


def create_artifact_id(relative_path: Path) -> str:
    """
    artifact_id identifies the physical file, and must be unique even when
    several files represent the same logical document.
    """
    base = relative_path.as_posix()

    # A short path hash, so that two unusual paths normalising to the same text
    # still come out distinct.
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8].upper()

    return f"{normalize_id(base)}_{digest}"


def assign_document_ids(records: list[dict]) -> list[dict]:
    """
    Give each record its document_id, appending the content role only where a
    base id is genuinely shared by different kinds of document.

    The previous version appended the role unconditionally, which produced two
    kinds of noise: it split the EU GMP annexes apart from nothing (see
    detect_content_role), and it decorated ids that had no sibling to be
    distinguished from — EMA_EMA_GMP_GDP_QA_Q_A, FDA_FDA_CGMP_QA_Q_A. Suffixing
    on demand leaves ICH_ICH_Q7 / ICH_ICH_Q7_Q_A, which is the case the suffix
    exists for, and nothing else.
    """
    roles_per_base: dict[str, set[str]] = {}
    for record in records:
        roles_per_base.setdefault(record["_base_id"], set()).add(record["content_role"])

    for record in records:
        base = record["_base_id"]
        role = record["content_role"]

        if len(roles_per_base[base]) > 1 and role != "primary":
            record["document_id"] = normalize_id(f"{base}_{role}")
        else:
            record["document_id"] = base

    return records


# ── Enrichment ───────────────────────────────────────────────────────────────
URL_IN_MARKDOWN = re.compile(r"^\s*(?:source|url)\s*:\s*(https?://\S+)", re.IGNORECASE | re.MULTILINE)


def harvest_url(path: Path) -> str | None:
    """
    Recover the official URL from inside the file.

    Every HTML page the downloaders saved carries a canonical link, and the
    generated Markdown and JSON record the page they were built from. That is
    13 of the 14 non-PDF artifacts resolved for free — and a real URL is the one
    thing an answer needs that it must never invent.

    PDFs carry nothing usable, so those stay for a human (or for the download
    scripts, which know the URL at the moment they fetch it) to fill in. Because
    document_url is a curated column, whatever gets typed there survives every
    later rebuild.
    """
    suffix = path.suffix.lower()

    try:
        if suffix in {".html", ".htm"}:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(
                path.read_text(encoding="utf-8", errors="replace"), "html.parser"
            )
            link = soup.find("link", rel=lambda value: value and "canonical" in value)
            if link and link.get("href"):
                return link["href"]

            for property_name in ("og:url", "citation_public_url"):
                meta = soup.find("meta", property=property_name) or soup.find(
                    "meta", attrs={"name": property_name}
                )
                if meta and meta.get("content"):
                    return meta["content"]

        elif suffix in {".md", ".txt"}:
            match = URL_IN_MARKDOWN.search(
                path.read_text(encoding="utf-8", errors="replace")
            )
            if match:
                return match.group(1).rstrip(").,")

        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            records = data if isinstance(data, list) else data.get("records", [])
            for record in records[:5] if isinstance(records, list) else []:
                if not isinstance(record, dict):
                    continue
                for key in ("source_url", "source", "url", "link"):
                    value = record.get(key)
                    if isinstance(value, str) and value.startswith("http"):
                        return value
    except Exception:                            # noqa: BLE001 - best effort only
        return None

    return None


def pdf_facts(path: Path) -> tuple[int | None, bool]:
    """
    (page_count, needs_ocr) for a PDF.

    needs_ocr is a fact worth recording rather than guessing at index time: a
    scanned guideline with no text layer extracts to nothing, and silently
    indexing an empty document is how a knowledge base ends up with a hole in
    it that nobody notices.
    """
    try:
        import pymupdf
    except ImportError:
        return None, False

    try:
        with pymupdf.open(path) as document:
            pages = document.page_count
            sample = min(pages, 5)
            has_text = any(
                (document[index].get_text() or "").strip() for index in range(sample)
            )
            return pages, (pages > 0 and not has_text)
    except Exception:                            # noqa: BLE001
        return None, False


# ── Scan ─────────────────────────────────────────────────────────────────────
def scan_documents(enrich: bool = True) -> tuple[list[dict], list[str]]:
    """Walk the knowledge base. Returns (records, unsupported_files)."""
    records: list[dict] = []
    unsupported: list[str] = []

    for path in sorted(KNOWLEDGE_BASE_DIR.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(KNOWLEDGE_BASE_DIR)

        if "_registry" in relative_path.parts:
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            unsupported.append(relative_path.as_posix())
            continue

        config = detect_source_config(relative_path)

        document_type = detect_document_type(relative_path)
        content_role = detect_content_role(relative_path)
        reference_code = detect_reference_code(relative_path)

        file_type = path.suffix.lower().lstrip(".")

        title = title_from_filename(path)
        if enrich and (title.lower() in JUNK_TITLES or len(title) < 4):
            title = title_from_content(path) or title

        page_count, needs_ocr = (None, False)
        if enrich and file_type == "pdf":
            page_count, needs_ocr = pdf_facts(path)

        records.append({
            "_base_id": create_document_base_id(relative_path, reference_code),
            "document_id": None,                 # assigned in assign_document_ids
            "artifact_id": create_artifact_id(relative_path),
            "document_title": title,
            "authority": config["authority"],
            "jurisdiction": config["jurisdiction"],
            "source_family": config["source_family"],
            "document_type": document_type,
            "content_role": content_role,
            "reference_code": reference_code,
            "topic": detect_topic(relative_path),
            "version": None,
            "publication_date": None,
            "effective_date": None,
            "status": "effective",
            "regulatory_weight": detect_regulatory_weight(
                document_type,
                config["source_family"],
            ),
            "language": "en",
            "file_type": file_type,
            "local_path": relative_path.as_posix(),
            "source_page_url": None,
            "document_url": harvest_url(path) if enrich else None,
            "official_source": True,
            "page_count": page_count,
            "text_extracted": path.suffix.lower()
            in {".json", ".md", ".txt", ".html", ".htm"},
            "needs_ocr": needs_ocr,
            "supersedes_document_id": None,
            "is_preferred_source": False,
            "ingest": True,
            "notes": None,
        })

    assign_document_ids(records)
    for record in records:
        record.pop("_base_id", None)

    return records, unsupported


def propagate_urls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Share a harvested URL across the artifacts of one logical document.

    When 21_CFR_Part_211 is on disk as .html, .json and .md, whichever of them
    yielded a URL speaks for all three — they are the same regulation.
    """
    df = df.copy()

    for document_id, group in df.groupby("document_id"):
        known = group["document_url"].dropna()
        known = [value for value in known if str(value).strip()]
        if not known:
            continue
        for index in group.index:
            if not str(df.at[index, "document_url"] or "").strip():
                df.at[index, "document_url"] = known[0]

    return df


# ── Preferred representation ─────────────────────────────────────────────────
def assign_preferred_sources(df: pd.DataFrame) -> pd.DataFrame:
    """
    Choose exactly one indexing representation per logical document.

        FDA_21_CFR_PART_210 .json / .html / .md

    all describe one regulation; only the highest-priority format is indexed.
    """
    df = df.copy()

    df["is_preferred_source"] = False
    df["format_priority"] = (
        df["file_type"].map(FORMAT_PRIORITY).fillna(0).astype(int)
    )

    for _document_id, group in df.groupby("document_id"):
        preferred_index = (
            group
            .sort_values(
                by=["format_priority", "local_path"],
                ascending=[False, True],
            )
            .index[0]
        )
        df.loc[preferred_index, "is_preferred_source"] = True

    return df.drop(columns=["format_priority"])


# ── Curation merge ───────────────────────────────────────────────────────────
def _is_blank(value) -> bool:
    """None, NaN, empty string or the string "nan" — all mean "not filled in"."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"nan", "none", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def merge_curation(
    fresh: pd.DataFrame,
    existing: pd.DataFrame | None,
    snapshot: pd.DataFrame | None,
) -> tuple[pd.DataFrame, int]:
    """
    Carry hand-entered values forward onto a freshly scanned registry.

    Matched on artifact_id, which is stable for as long as the file stays where
    it is.

    The hard part is telling a *curated* value from a value this script derived
    on an earlier run. "Preserve anything non-blank" is not good enough: on the
    first run after the title logic improved, it would lovingly preserve the old
    derived title "here" — the literal word from 08_here.html — and the
    improvement would never reach the registry. Worse, it cannot be inverted:
    "only preserve where the fresh value is blank" would silently reset a
    hand-set `status = superseded` back to `effective`, which is the single most
    dangerous thing this file can get wrong.

    So each run records what it derived, in documents_registry.derived.csv. A
    value is curation when it differs from what was derived last time, and is
    kept. A value equal to the old derivation is just an old guess, and gives
    way to the new one.

    With no snapshot yet (the first run after this change), anything non-blank
    is treated as curation — the cautious reading. Use --rebuild once to start
    from clean derivation.
    """
    # object dtype throughout: the scanned frame types `ingest` as bool and
    # `page_count` as int, while everything read back from the CSV is a string,
    # and pandas refuses the assignment rather than widening the column.
    fresh = fresh.astype(object)

    if existing is None or existing.empty or "artifact_id" not in existing.columns:
        return fresh, 0

    previous = _index_by_artifact(existing)
    derived_before = _index_by_artifact(snapshot)
    preserved = 0

    for position, artifact_id in enumerate(fresh["artifact_id"]):
        old_row = previous.get(artifact_id)
        if old_row is None:
            continue
        was_derived = derived_before.get(artifact_id, {})

        for column in CURATED_COLUMNS:
            old_value = old_row.get(column)
            if _is_blank(old_value):
                continue
            if str(old_value) == str(fresh.iloc[position][column]):
                continue
            if column in was_derived and str(old_value) == str(was_derived[column]):
                continue                          # an old derivation, not curation
            fresh.iat[position, fresh.columns.get_loc(column)] = old_value
            preserved += 1

    return fresh, preserved


def _index_by_artifact(frame: pd.DataFrame | None) -> dict[str, dict]:
    if frame is None or frame.empty or "artifact_id" not in frame.columns:
        return {}
    return {
        str(row["artifact_id"]): row
        for row in frame.to_dict(orient="records")
    }


# ── Validation ───────────────────────────────────────────────────────────────
def validate_registry(df: pd.DataFrame) -> list[str]:
    """Structural problems that make the registry unsafe to index from."""
    errors: list[str] = []

    duplicate_artifacts = df[df["artifact_id"].duplicated(keep=False)]
    if not duplicate_artifacts.empty:
        errors.append(
            "Duplicate artifact_id values detected:\n"
            + duplicate_artifacts[["artifact_id", "local_path"]].to_string(index=False)
        )

    # Exactly one preferred representation per logical document.
    for document_id, group in df.groupby("document_id"):
        preferred_count = int(group["is_preferred_source"].astype(bool).sum())
        if preferred_count != 1:
            errors.append(
                f"{document_id}: expected exactly one preferred source, "
                f"found {preferred_count}"
            )

    # Two live rows of the same format under one document_id cannot be
    # alternate representations, so they are two different documents wearing
    # one id. That has to be looked at rather than guessed about.
    duplicates = df[df["document_id"].duplicated(keep=False)]
    for document_id, group in duplicates.groupby("document_id"):
        counts = group["file_type"].value_counts()
        if (counts > 1).any():
            errors.append(
                f"\nAmbiguous logical document: {document_id}\n"
                "Several artifacts share one file type, so they cannot be "
                "alternate representations of the same document.\n\n"
                + group[
                    ["document_id", "document_title", "content_role",
                     "file_type", "local_path"]
                ].to_string(index=False)
            )

    return errors


# ── Reporting ────────────────────────────────────────────────────────────────
def print_duplicate_summary(df: pd.DataFrame) -> None:
    duplicated = df[df["document_id"].duplicated(keep=False)]

    if duplicated.empty:
        print("\nNo alternate document representations found.")
        return

    print("\nAlternate representations:")
    print("-" * 80)

    for document_id, group in duplicated.groupby("document_id"):
        print(f"\n{document_id}")
        for _, row in group.iterrows():
            marker = " <-- INDEXED" if row["is_preferred_source"] else ""
            print(f"  {row['file_type']:5} {row['local_path']}{marker}")


def print_gaps(df: pd.DataFrame, unsupported: list[str]) -> None:
    indexed = df[df["ingest"].astype(bool) & df["is_preferred_source"].astype(bool)]

    missing_url = indexed[
        indexed["document_url"].isna()
        & indexed["source_page_url"].isna()
    ]

    print("\nCuration gaps (indexed documents only):")
    print("-" * 80)
    print(f"  no official URL       : {len(missing_url)} of {len(indexed)}")
    for column in ("version", "publication_date", "effective_date"):
        print(f"  no {column:<19}: {int(indexed[column].isna().sum())} of {len(indexed)}")

    if len(missing_url):
        print("\n  An answer citing these can quote them but cannot link to the")
        print("  official source. document_url is a curated column — fill it in")
        print("  once and rebuilds will keep it.")

    needs_ocr = indexed[indexed["needs_ocr"].astype(bool)]
    if len(needs_ocr):
        print(f"\n  ! {len(needs_ocr)} PDF(s) have no extractable text layer:")
        for path in needs_ocr["local_path"].head(10):
            print(f"      {path}")

    if unsupported:
        print(f"\n  ! {len(unsupported)} file(s) in KnowledgeBase/ have an unsupported")
        print("    extension and are not registered, so they are never indexed:")
        for path in unsupported[:10]:
            print(f"      {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the AskGMP document registry from KnowledgeBase/."
    )
    parser.add_argument("--check", action="store_true",
                        help="Validate and report, but write nothing.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Discard curated values instead of carrying them "
                             "forward. Destroys hand-entered URLs and dates.")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip opening files for URLs, titles and page counts.")
    return parser.parse_args()


REGISTRY_CSV     = OUTPUT_DIR / "documents_registry.csv"
REGISTRY_PARQUET = OUTPUT_DIR / "documents_registry.parquet"

# What the last run derived, before any curation was merged in. Comparing
# against it is what lets a rebuild tell "someone typed this" from "an older
# version of this script guessed this".
DERIVED_SNAPSHOT = OUTPUT_DIR / "documents_registry.derived.csv"


def _read_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=object)
    except Exception as exc:                     # noqa: BLE001
        print(f"  ! Could not read {label} ({exc}); continuing without it.")
        return None


def load_existing() -> pd.DataFrame | None:
    return _read_csv(REGISTRY_CSV, "the existing registry")


def load_snapshot() -> pd.DataFrame | None:
    return _read_csv(DERIVED_SNAPSHOT, "the derived snapshot")


def main() -> int:
    args = parse_args()

    if not KNOWLEDGE_BASE_DIR.is_dir():
        print(f"Knowledge base folder not found: {KNOWLEDGE_BASE_DIR.resolve()}")
        print("Run this from the project root.")
        return 1

    records, unsupported = scan_documents(enrich=not args.no_enrich)
    if not records:
        print(f"No supported documents found under {KNOWLEDGE_BASE_DIR}.")
        return 1

    df = pd.DataFrame(records).reindex(columns=DOCUMENT_COLUMNS)
    df = propagate_urls(df)
    df = assign_preferred_sources(df)

    derived = df.copy()                      # pre-merge: what this run derived

    preserved = 0
    if not args.rebuild:
        df, preserved = merge_curation(df, load_existing(), load_snapshot())
        # is_preferred_source is derived, so it is recomputed after the merge
        # in case a curated ingest flag changed which rows are eligible.
        df = assign_preferred_sources(df)

    df = df.sort_values(
        by=["source_family", "document_id", "local_path"],
        na_position="last",
    ).reset_index(drop=True)

    errors = validate_registry(df)

    print("=" * 80)
    print("ASKGMP — Document Registry")
    print("=" * 80)
    print(f"Physical artifacts             : {len(df)}")
    print(f"Logical documents              : {df['document_id'].nunique()}")
    print(f"Preferred representations      : {int(df['is_preferred_source'].astype(bool).sum())}")
    print(f"Selected for ingestion         : "
          f"{int((df['ingest'].astype(bool) & df['is_preferred_source'].astype(bool)).sum())}")
    if preserved:
        print(f"Curated values carried forward : {preserved}")

    print("\nDocuments per source:")
    print(df["source_family"].value_counts(dropna=False).to_string())

    print_duplicate_summary(df)
    print_gaps(df, unsupported)

    if errors:
        print("\n" + "=" * 80)
        print("REGISTRY VALIDATION FAILED — nothing written")
        print("=" * 80)
        for error in errors:
            print(error)
        return 1

    if args.check:
        print("\n--check: registry is valid. Nothing written.")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # A page count of "29.0" is pandas widening the column to float because
    # some rows are empty, not a fact about the document.
    df["page_count"] = pd.to_numeric(df["page_count"], errors="coerce").astype("Int64")
    derived["page_count"] = pd.to_numeric(
        derived["page_count"], errors="coerce"
    ).astype("Int64")

    df.to_csv(REGISTRY_CSV, index=False, encoding="utf-8-sig")
    derived.to_csv(DERIVED_SNAPSHOT, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(REGISTRY_PARQUET, index=False)
    except Exception as exc:                     # noqa: BLE001 - pyarrow optional
        print(f"\n  ! Parquet not written ({exc}). The CSV is the file that matters.")

    print(f"\nCSV:      {REGISTRY_CSV}")
    print(f"Parquet:  {REGISTRY_PARQUET}")
    print(f"Snapshot: {DERIVED_SNAPSHOT}  (derivation record; do not hand-edit)")

    print("\nPreview:")
    print(
        df[[
            "document_id", "document_title", "authority", "document_type",
            "content_role", "reference_code", "file_type",
            "is_preferred_source", "ingest",
        ]].head(20).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
