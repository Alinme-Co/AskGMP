"""
Tests for the AskGMP knowledge base indexer.

Nothing here touches OpenAI or Qdrant. The heavy extraction libraries
(pymupdf, python-pptx) are stubbed through sys.modules, so the fallback paths
can be driven deliberately rather than waited for.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app.documents import cache as extraction_cache
from app.documents import chunking
from app.documents import extractors
from app.knowledge_base import indexer as kbi
from app.knowledge_base import registry as registry_module
from app.knowledge_base.registry import (
    discover_documents,
    load_registry,
    resolve_duplicates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
REGISTRY_HEADER = ",".join(registry_module.REGISTRY_COLUMNS)


def registry_row(**overrides) -> dict:
    """
    One registry row. artifact_id defaults to something derived from
    local_path, mirroring the real registry where it is unique per file.
    """
    row = {column: "" for column in registry_module.REGISTRY_COLUMNS}
    row.update({
        "document_id"      : "ICH_Q9",
        "document_title"   : "ICH Q9(R1) Quality Risk Management",
        "content_role"     : "primary",
        "is_preferred_source": "True",
        "authority"        : "ICH",
        "jurisdiction"     : "International",
        "source_family"    : "ich",
        "document_type"    : "guideline",
        "reference_code"   : "ICH Q9(R1)",
        "topic"            : "quality risk management",
        "status"           : "effective",
        "regulatory_weight": "official_guideline",
        "language"         : "en",
        "file_type"        : "pdf",
        "local_path"       : "ich/q9/ICH_Q9.pdf",
        "document_url"     : "https://database.ich.org/sites/default/files/ICH_Q9.pdf",
        "official_source"  : "True",
        "ingest"           : "True",
    })
    row.update(overrides)
    row.setdefault("artifact_id", "")
    if not row["artifact_id"]:
        stem = Path(str(row["local_path"])).stem.upper()
        suffix = Path(str(row["local_path"])).suffix.lstrip(".").upper()
        row["artifact_id"] = f"{stem}_{suffix}_ARTIFACT"
    return row


def write_registry(tmp_path: Path, rows: list[dict]) -> Path:
    registry_dir = tmp_path / "KnowledgeBase" / "_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / "documents_registry.csv"

    lines = [REGISTRY_HEADER]
    for row in rows:
        lines.append(",".join(
            '"' + str(row.get(column, "")).replace('"', '""') + '"'
            for column in registry_module.REGISTRY_COLUMNS
        ))
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def touch(tmp_path: Path, local_path: str, content: str = "x") -> Path:
    path = tmp_path / "KnowledgeBase" / local_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def knowledge_base(tmp_path: Path) -> Path:
    return tmp_path / "KnowledgeBase"


# ── 1. Registry loading ───────────────────────────────────────────────────────
def test_registry_loads_columns_and_normalises_values(tmp_path):
    path = write_registry(tmp_path, [registry_row(topic="", page_count="")])
    rows = load_registry(path)

    assert len(rows) == 1
    row = rows[0]

    assert row["document_id"] == "ICH_Q9"
    assert row["authority"] == "ICH"
    assert row["reference_code"] == "ICH Q9(R1)"
    # Empty cells become None, not "" or NaN — a NaN is not JSON-serialisable
    # into a Qdrant payload and not filterable once it is there.
    assert row["topic"] is None
    assert row["page_count"] is None
    assert row["ingest"] is True
    assert row["official_source"] is True


def test_registry_tolerates_a_missing_column(tmp_path):
    """A registry written by an older builder still loads."""
    path = tmp_path / "reg.csv"
    path.write_text(
        "document_id,document_title,local_path,ingest\n"
        "A,Title A,ich/a.pdf,True\n",
        encoding="utf-8-sig",
    )
    rows = load_registry(path)
    assert rows[0]["document_id"] == "A"
    assert rows[0]["supersedes_document_id"] is None


def test_missing_registry_file_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "nope.csv")


# ── 2. ingest = False is never indexed ────────────────────────────────────────
@pytest.mark.parametrize("flag", ["False", "false", "0", "no", ""])
def test_ingest_false_is_excluded(tmp_path, knowledge_base, flag):
    path = write_registry(tmp_path, [
        registry_row(document_id="KEEP", local_path="ich/keep.pdf", ingest="True"),
        registry_row(document_id="DROP", local_path="ich/drop.pdf", ingest=flag),
    ])
    touch(tmp_path, "ich/keep.pdf")
    touch(tmp_path, "ich/drop.pdf")

    discovery = discover_documents(knowledge_base, path)

    assert [entry["document_id"] for entry in discovery.documents] == ["KEEP"]
    assert len(discovery.not_ingested) == 1
    # The file exists — it is excluded by policy, not by absence.
    assert not discovery.missing


# ── 3. Missing files warn and skip, never crash ───────────────────────────────
def test_missing_file_is_reported_and_skipped(tmp_path, knowledge_base):
    path = write_registry(tmp_path, [
        registry_row(document_id="PRESENT", local_path="ich/present.pdf"),
        registry_row(document_id="ABSENT", local_path="ich/absent.pdf"),
    ])
    touch(tmp_path, "ich/present.pdf")

    discovery = discover_documents(knowledge_base, path)

    assert [entry["document_id"] for entry in discovery.documents] == ["PRESENT"]
    assert [row["document_id"] for row in discovery.missing] == ["ABSENT"]
    assert discovery.missing[0]["problem"] == "file not found"


def test_unsupported_format_is_skipped_not_crashed(tmp_path, knowledge_base):
    path = write_registry(tmp_path, [
        registry_row(document_id="BIN", local_path="ich/thing.bin", file_type="bin"),
    ])
    touch(tmp_path, "ich/thing.bin")

    discovery = discover_documents(knowledge_base, path)

    assert not discovery.documents
    assert "unsupported format" in discovery.missing[0]["problem"]


def test_unregistered_files_are_found_but_not_indexed(tmp_path, knowledge_base):
    path = write_registry(tmp_path, [
        registry_row(document_id="KNOWN", local_path="ich/known.pdf"),
    ])
    touch(tmp_path, "ich/known.pdf")
    touch(tmp_path, "ich/stranger.pdf")

    discovery = discover_documents(knowledge_base, path)

    assert [entry["document_id"] for entry in discovery.documents] == ["KNOWN"]
    assert discovery.unregistered == ["ich/stranger.pdf"]


# ── 4. Duplicate document_id: representation selection ────────────────────────
def test_alternate_formats_collapse_to_the_most_structured(tmp_path, knowledge_base):
    """json > md > html for one logical document held three ways."""
    rows = [
        registry_row(document_id="FDA_211", local_path="fda/211.html", file_type="html"),
        registry_row(document_id="FDA_211", local_path="fda/211.json", file_type="json"),
        registry_row(document_id="FDA_211", local_path="fda/211.md",   file_type="md"),
    ]
    path = write_registry(tmp_path, rows)
    for row in rows:
        touch(tmp_path, row["local_path"])

    discovery = discover_documents(knowledge_base, path)

    assert len(discovery.documents) == 1
    assert discovery.documents[0]["local_path"] == "fda/211.json"

    skipped = {row["local_path"] for row in discovery.skipped_representations}
    assert skipped == {"fda/211.html", "fda/211.md"}
    assert all(row["superseded_by_path"] == "fda/211.json"
               for row in discovery.skipped_representations)


def test_same_format_collision_keeps_both_documents(tmp_path, knowledge_base):
    """
    ICH_Q7 is both the guideline and its separate Q&A, and the reference-code
    ID collides. Both are real content; dropping one would lose guidance.
    """
    rows = [
        registry_row(document_id="ICH_Q7", local_path="ich/q7/Q7_Guide.pdf"),
        registry_row(document_id="ICH_Q7", local_path="ich/q7/Q7_QA.pdf"),
    ]
    path = write_registry(tmp_path, rows)
    for row in rows:
        touch(tmp_path, row["local_path"])

    discovery = discover_documents(knowledge_base, path)

    assert len(discovery.documents) == 2
    assert not discovery.skipped_representations
    assert all(entry["id_collision"] for entry in discovery.documents)

    # artifact_id is unique per file, so it is the unit key.
    keys = sorted(entry["document_key"] for entry in discovery.documents)
    assert keys == ["Q7_GUIDE_PDF_ARTIFACT", "Q7_QA_PDF_ARTIFACT"]
    # The registry's document_id is left exactly as recorded.
    assert {entry["document_id"] for entry in discovery.documents} == {"ICH_Q7"}


def test_collision_without_artifact_id_falls_back_to_the_path_stem(tmp_path, knowledge_base):
    """A registry written before artifact_id existed still disambiguates."""
    rows = [
        registry_row(document_id="ICH_Q7", local_path="ich/q7/Q7_Guide.pdf", artifact_id=""),
        registry_row(document_id="ICH_Q7", local_path="ich/q7/Q7_QA.pdf", artifact_id=""),
    ]
    for row in rows:
        row["artifact_id"] = ""            # undo the fixture default
        touch(tmp_path, row["local_path"])
    path = write_registry(tmp_path, rows)

    discovery = discover_documents(knowledge_base, path)

    keys = sorted(entry["document_key"] for entry in discovery.documents)
    assert keys == ["ICH_Q7::Q7_Guide", "ICH_Q7::Q7_QA"]


def test_resolve_duplicates_leaves_unique_ids_alone():
    rows = [
        {"document_id": "A", "local_path": "a.pdf"},
        {"document_id": "B", "local_path": "b.pdf"},
    ]
    kept, skipped = resolve_duplicates(rows)
    assert len(kept) == 2 and not skipped
    assert [row["document_key"] for row in kept] == ["A", "B"]


# ── 5. JSON Q&A extraction ────────────────────────────────────────────────────
def test_json_qa_becomes_one_atomic_record_each(tmp_path):
    payload = [
        {
            "section" : "Chapter 1: Pharmaceutical quality system",
            "question": "What should be the frequency of the product quality review?",
            "answer"  : "The product review is expected annually.",
            "source"  : "https://www.ema.europa.eu/en/qa",
        },
        {
            "section" : "Chapter 3: Equipment",
            "question": "Should metal detectors be used routinely?",
            "answer"  : "It is recommended for processes prone to this.",
            "source"  : "https://www.ema.europa.eu/en/qa",
        },
    ]
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, status = extractors.extract_records(str(path))

    assert len(records) == 2, "the file must not collapse into one blob"
    assert "q&a" in status

    first = records[0]
    assert first["atomic"] is True
    assert "## Question" in first["text"] and "## Answer" in first["text"]
    assert "frequency of the product quality review" in first["text"]
    assert first["section"] == "Chapter 1: Pharmaceutical quality system"
    assert first["meta"]["question"].startswith("What should be")
    assert first["meta"]["qa_section"] == "Chapter 1: Pharmaceutical quality system"
    assert first["meta"]["record_url"] == "https://www.ema.europa.eu/en/qa"


def test_json_section_records_keep_their_own_urls(tmp_path):
    """21 CFR structured JSON: one record per section, each with its eCFR URL."""
    payload = [
        {
            "section"   : "211.100",
            "heading"   : "Written procedures; deviations.",
            "text"      : "There shall be written procedures for production.",
            "source_url": "https://www.ecfr.gov/.../section-211.100",
        },
    ]
    path = tmp_path / "cfr.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = extractors.extract_records(str(path))

    assert len(records) == 1
    assert records[0]["atomic"] is True
    assert records[0]["section"] == "211.100 Written procedures; deviations."
    assert records[0]["meta"]["record_url"].endswith("section-211.100")


def test_non_qa_json_falls_back_to_readable_prose(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"authority": "WHO", "scope": "Water systems"}),
                    encoding="utf-8")

    records, status = extractors.extract_records(str(path))

    assert len(records) == 1
    assert "unstructured" in status
    assert "Authority: WHO" in records[0]["text"]


# ── 6. PDF extraction fallback ────────────────────────────────────────────────
class _FakePage:
    def __init__(self, text):
        self._text = text

    def get_text(self, *args, **kwargs):
        return self._text


class _FakeDocument:
    def __init__(self, pages):
        self._pages = [_FakePage(text) for text in pages]
        self.page_count = len(pages)

    def __getitem__(self, index):
        return self._pages[index]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_pymupdf(monkeypatch):
    """Install stub pymupdf / pymupdf4llm modules and hand back a control knob."""
    control = {"pages": ["Page one text", "Page two text"], "markdown": None}

    pymupdf = types.ModuleType("pymupdf")
    pymupdf.open = lambda path: _FakeDocument(control["pages"])
    monkeypatch.setitem(sys.modules, "pymupdf", pymupdf)

    pymupdf4llm = types.ModuleType("pymupdf4llm")

    def to_markdown(path, **kwargs):
        if control["markdown"] is None:
            raise RuntimeError("markdown extraction exploded")
        return control["markdown"]

    pymupdf4llm.to_markdown = to_markdown
    monkeypatch.setitem(sys.modules, "pymupdf4llm", pymupdf4llm)

    return control


def test_pdf_falls_back_to_plain_pymupdf_when_markdown_fails(fake_pymupdf):
    records, status = extractors.extract_records("whatever.pdf", ".pdf")

    assert status.startswith("markdown-fallback")
    assert [record["text"] for record in records] == ["Page one text", "Page two text"]
    assert [record["page"] for record in records] == [1, 2]


def test_pdf_fills_individual_empty_pages_from_the_fallback(fake_pymupdf):
    """
    A page over the graphics limit comes back empty from pymupdf4llm rather
    than raising. That page — and only that page — is refilled.
    """
    fake_pymupdf["markdown"] = [
        {"metadata": {"page": 1}, "text": "# Real markdown page one"},
        {"metadata": {"page": 2}, "text": "   "},
    ]
    fake_pymupdf["pages"] = ["ignored", "recovered page two"]

    records, status = extractors.extract_records("whatever.pdf", ".pdf")

    assert records[0]["text"] == "# Real markdown page one"
    assert records[1]["text"] == "recovered page two"
    assert "fallback" in status


def test_pdf_with_no_text_anywhere_yields_nothing(fake_pymupdf):
    fake_pymupdf["pages"] = ["", "   "]
    records, _ = extractors.extract_records("whatever.pdf", ".pdf")
    assert records == []


# ── 7. PPTX extraction ────────────────────────────────────────────────────────
class _FakeTextFrame:
    def __init__(self, text):
        self.text = text


class _FakeShape:
    shape_type = 1

    def __init__(self, text):
        self.has_text_frame = True
        self.text_frame = _FakeTextFrame(text)


class _FakeShapes(list):
    def __init__(self, shapes, title=None):
        super().__init__(shapes)
        self.title = title


class _FakeSlide:
    def __init__(self, title, bodies, notes=None):
        title_shape = _FakeShape(title) if title else None
        shapes = ([title_shape] if title_shape else []) + [_FakeShape(b) for b in bodies]
        self.shapes = _FakeShapes(shapes, title=title_shape)
        self.has_notes_slide = notes is not None
        if notes is not None:
            self.notes_slide = types.SimpleNamespace(
                notes_text_frame=_FakeTextFrame(notes)
            )


@pytest.fixture
def fake_pptx(monkeypatch):
    slides = []
    module = types.ModuleType("pptx")
    module.Presentation = lambda path: types.SimpleNamespace(slides=slides)
    monkeypatch.setitem(sys.modules, "pptx", module)
    return slides


def test_pptx_makes_one_atomic_record_per_slide(fake_pptx):
    fake_pptx.extend([
        _FakeSlide("Out of Specification Investigations",
                   ["Phase I laboratory investigation", "Phase II full investigation"],
                   notes="Remember the retest rules."),
        _FakeSlide("Invalidating a Result", ["Only with an assignable cause"]),
    ])

    records, status = extractors.extract_records("deck.pptx", ".pptx")

    assert status == "extracted"
    assert len(records) == 2

    first = records[0]
    assert first["page"] == 1                       # slide number acts as page
    assert first["atomic"] is True
    assert first["section"] == "Slide 1 — Out of Specification Investigations"
    assert "Out of Specification Investigations" in first["text"]
    assert "Phase I laboratory investigation" in first["text"]
    assert "Speaker notes: Remember the retest rules." in first["text"]

    assert records[1]["page"] == 2
    assert "Only with an assignable cause" in records[1]["text"]


def test_pptx_skips_empty_slides(fake_pptx):
    fake_pptx.extend([_FakeSlide(None, []), _FakeSlide("Real slide", ["Content"])])

    records, _ = extractors.extract_records("deck.pptx", ".pptx")

    assert len(records) == 1
    assert records[0]["page"] == 2


# ── 8. Metadata-sensitive fingerprinting ──────────────────────────────────────
def _entry(**overrides) -> dict:
    entry = {
        "document_key"  : "ICH_Q9",
        "document_id"   : "ICH_Q9",
        "document_title": "ICH Q9(R1) Quality Risk Management",
        "authority"     : "ICH",
        "jurisdiction"  : "International",
        "source_family" : "ich",
        "document_type" : "guideline",
        "reference_code": "ICH Q9(R1)",
        "topic"         : "quality risk management",
        "version"       : "R1",
        "publication_date": "2023-01-18",
        "effective_date": "2023-01-18",
        "status"        : "effective",
        "regulatory_weight": "official_guideline",
        "source_page_url": None,
        "document_url"  : "https://database.ich.org/ICH_Q9.pdf",
    }
    entry.update(overrides)
    return entry


def test_fingerprint_changes_when_status_changes_but_file_does_not():
    """
    The case this whole design exists for: a guideline is superseded, the PDF on
    disk is byte-identical, and the index must still be rebuilt.
    """
    file_hash = "a" * 64
    effective = kbi.document_fingerprint(_entry(status="effective"), file_hash)
    superseded = kbi.document_fingerprint(_entry(status="superseded"), file_hash)

    assert effective != superseded


@pytest.mark.parametrize("field,value", [
    ("document_title", "A different title"),
    ("authority", "EMA"),
    ("jurisdiction", "EU"),
    ("reference_code", "ICH Q9"),
    ("topic", "validation"),
    ("version", "R2"),
    ("effective_date", "2024-06-01"),
    ("regulatory_weight", "supporting_guidance"),
    ("document_url", "https://example.org/other.pdf"),
    ("source_page_url", "https://example.org/page"),
])
def test_fingerprint_tracks_every_payload_bearing_field(field, value):
    file_hash = "a" * 64
    baseline = kbi.document_fingerprint(_entry(), file_hash)
    assert kbi.document_fingerprint(_entry(**{field: value}), file_hash) != baseline


def test_fingerprint_changes_when_the_file_changes():
    entry = _entry()
    assert (kbi.document_fingerprint(entry, "a" * 64)
            != kbi.document_fingerprint(entry, "b" * 64))


def test_fingerprint_is_stable_for_identical_input():
    assert (kbi.document_fingerprint(_entry(), "a" * 64)
            == kbi.document_fingerprint(_entry(), "a" * 64))


def test_fingerprint_changes_with_the_chunk_budget(monkeypatch):
    """Re-chunking at a different budget must invalidate what is stored."""
    baseline = kbi.document_fingerprint(_entry(), "a" * 64)
    monkeypatch.setattr(kbi, "MAX_CHUNK_TOKENS", 8000)
    assert kbi.document_fingerprint(_entry(), "a" * 64) != baseline


# ── 9. The 1000-token chunk budget ────────────────────────────────────────────
def test_default_budget_is_1000_tokens():
    assert chunking.MAX_CHUNK_TOKENS == 1000


def test_chunks_stay_within_the_budget():
    paragraph = ("The quality control unit shall have the responsibility and "
                 "authority to approve or reject all components. ") * 12
    records = [
        {"text": paragraph, "page": page, "section": None, "atomic": False, "meta": {}}
        for page in range(1, 9)
    ]

    chunks = chunking.chunk_records(records, budget=1000)

    assert len(chunks) > 1
    assert all(chunking.token_count(chunk["text"]) <= 1000 for chunk in chunks)


def test_a_short_qa_is_never_split_or_merged():
    records = [
        {"text": "# Chapter 1\n\n## Question\nQ1?\n\n## Answer\nA1.",
         "page": None, "section": "Chapter 1", "atomic": True, "meta": {"question": "Q1?"}},
        {"text": "# Chapter 3\n\n## Question\nQ2?\n\n## Answer\nA2.",
         "page": None, "section": "Chapter 3", "atomic": True, "meta": {"question": "Q2?"}},
    ]

    chunks = chunking.chunk_records(records, budget=1000)

    # Two short Q&As together use a fraction of the budget. Filling the chunk
    # with the next Q&A would only make both harder to retrieve.
    assert len(chunks) == 2
    assert chunks[0]["meta"]["question"] == "Q1?"
    assert chunks[1]["meta"]["question"] == "Q2?"


def test_unrelated_sections_are_not_merged_to_fill_the_budget():
    records = [
        {"text": "Short text about premises.", "page": None,
         "section": "3. Premises", "atomic": False, "meta": {}},
        {"text": "Short text about equipment.", "page": None,
         "section": "4. Equipment", "atomic": False, "meta": {}},
    ]
    chunks = chunking.chunk_records(records, budget=1000)
    assert len(chunks) == 2


def test_consecutive_pages_of_one_section_do_merge():
    records = [
        {"text": "First half of the passage.", "page": 4,
         "section": None, "atomic": False, "meta": {}},
        {"text": "Second half of the passage.", "page": 5,
         "section": None, "atomic": False, "meta": {}},
    ]
    chunks = chunking.chunk_records(records, budget=1000)

    assert len(chunks) == 1
    assert chunks[0]["page_start"] == 4 and chunks[0]["page_end"] == 5
    assert chunks[0]["section"] == "pp. 4-5"


def test_oversized_text_splits_on_structure_first():
    text = "\n\n".join(f"## Section {n}\n\n" + ("word " * 400) for n in range(1, 5))
    parts = chunking.split_text(text, budget=1000)

    assert len(parts) >= 4
    assert all(chunking.token_count(part) <= 1000 for part in parts)
    assert parts[0].startswith("## Section 1")


def test_an_unsplittable_blob_is_returned_rather_than_looping():
    parts = chunking.split_text("x" * 20000, budget=50)
    assert parts == ["x" * 20000]


# ── 10. The Qdrant payload ────────────────────────────────────────────────────
def _chunk(**overrides) -> dict:
    chunk = {
        "text"      : "There shall be written procedures for production and process control.",
        "page_start": 12,
        "page_end"  : 13,
        "section"   : "5. Production (pp. 12-13)",
        "meta"      : {},
    }
    chunk.update(overrides)
    return chunk


def test_payload_carries_citation_and_regulatory_metadata():
    entry = dict(_entry(), local_path="ich/q9/ICH_Q9.pdf", source_file="ICH_Q9.pdf",
                 language="en", file_type="pdf", official_source=True,
                 supersedes_document_id="ICH_Q9_R0", notes=None)

    payload = kbi.build_payload(entry, _chunk(), 3, "ICH_Q9#chunk-0003")

    # Citation essentials.
    assert payload["document_title"] == "ICH Q9(R1) Quality Risk Management"
    assert payload["reference_code"] == "ICH Q9(R1)"
    assert payload["authority"] == "ICH"
    assert payload["document_url"] == "https://database.ich.org/ICH_Q9.pdf"
    assert payload["citation_url"] == "https://database.ich.org/ICH_Q9.pdf"
    assert payload["section"] == "5. Production (pp. 12-13)"
    assert payload["page"] == 12 and payload["page_end"] == 13

    # Regulatory state, preserved exactly.
    assert payload["status"] == "effective"
    assert payload["effective_date"] == "2023-01-18"
    assert payload["supersedes_document_id"] == "ICH_Q9_R0"
    assert payload["regulatory_weight"] == "official_guideline"
    assert payload["jurisdiction"] == "International"

    # AskGMP semantics — emphatically not doc_type="sop".
    assert payload["doc_type"] == "regulation"
    assert payload["source"] == "askgmp"

    assert payload["chunk_no"] == 3
    assert payload["chunk_id"] == "ICH_Q9#chunk-0003"
    assert payload["source_path"] == "ich/q9/ICH_Q9.pdf"


def test_payload_keeps_a_non_effective_status_as_recorded():
    for status in ("superseded", "withdrawn", "draft", "published_not_yet_effective"):
        payload = kbi.build_payload(_entry(status=status), _chunk(), 0, "id#0")
        assert payload["status"] == status


def test_payload_prefers_a_record_level_url_for_the_citation():
    """An eCFR section URL beats the document-level one — it is more precise."""
    entry = _entry(document_url="https://www.ecfr.gov/part-211")
    chunk = _chunk(meta={"record_url": "https://www.ecfr.gov/part-211/section-211.100"})

    payload = kbi.build_payload(entry, chunk, 0, "id#0")

    assert payload["citation_url"].endswith("section-211.100")
    assert payload["record_url"].endswith("section-211.100")
    assert payload["document_url"] == "https://www.ecfr.gov/part-211"


def test_payload_falls_back_to_the_source_page_url():
    entry = _entry(document_url=None, source_page_url="https://www.ema.europa.eu/page")
    payload = kbi.build_payload(entry, _chunk(), 0, "id#0")
    assert payload["citation_url"] == "https://www.ema.europa.eu/page"


def test_payload_carries_qa_fields_when_present():
    chunk = _chunk(meta={
        "question"  : "What should be the frequency of the PQR?",
        "qa_section": "Chapter 1: Pharmaceutical quality system",
        "qa_topic"  : "General Provisions",
    })
    payload = kbi.build_payload(_entry(), chunk, 0, "id#0")

    assert payload["question"].startswith("What should be")
    assert payload["qa_section"].startswith("Chapter 1")
    assert payload["qa_topic"] == "General Provisions"


def test_payload_has_no_nan_values():
    """A NaN reaching Qdrant is neither serialisable nor filterable."""
    entry = _entry(topic=float("nan"), version=float("nan"))
    payload = kbi.build_payload(entry, _chunk(), 0, "id#0")

    assert payload["topic"] is None
    assert payload["version"] is None
    assert not any(
        isinstance(value, float) and value != value for value in payload.values()
    )


def test_missing_qa_fields_are_simply_absent():
    payload = kbi.build_payload(_entry(), _chunk(), 0, "id#0")
    assert "question" not in payload
    assert "qa_section" not in payload


# ── Titles ────────────────────────────────────────────────────────────────────
def test_registry_title_wins_over_document_text():
    entry = dict(_entry(), local_path="ich/q9/ICH_Q9.pdf")
    records = [{"text": "EUROPEAN COMMISSION\nSome other heading entirely"}]

    assert kbi.resolve_title(entry, records) == "ICH Q9(R1) Quality Risk Management"


def test_title_is_derived_only_when_the_registry_cell_is_blank():
    entry = dict(_entry(document_title=None), local_path="ich/q9/ICH_Q9.pdf")
    records = [{"text": "EUROPEAN COMMISSION\nBrussels\nQuality Risk Management"}]

    # The letterhead lines are skipped; the real heading is taken.
    assert kbi.resolve_title(entry, records) == "Quality Risk Management"


def test_title_falls_back_to_the_filename_when_there_is_nothing_else():
    entry = dict(_entry(document_title=None), local_path="ich/q9/ICH_Q9_Draft.pdf")
    assert kbi.resolve_title(entry, []) == "ICH Q9 Draft"


# ── Point IDs and sparse vectors ──────────────────────────────────────────────
def test_point_ids_are_deterministic_and_typed():
    from app.core.constants import DOC_TYPE_REGULATION, DOC_TYPE_SOP
    from app.infrastructure.qdrant import point_id

    assert point_id(DOC_TYPE_REGULATION, "A#0") == point_id(DOC_TYPE_REGULATION, "A#0")
    assert point_id(DOC_TYPE_REGULATION, "A#0") != point_id(DOC_TYPE_REGULATION, "A#1")
    # A regulation and an SOP with the same chunk id must not collide.
    assert point_id(DOC_TYPE_REGULATION, "A#0") != point_id(DOC_TYPE_SOP, "A#0")


def test_sparse_vectors_are_normalised_and_shared_with_the_query_side():
    from app.infrastructure.embeddings import build_sparse_terms

    indices, values = build_sparse_terms("data integrity data integrity ALCOA")

    assert len(indices) == len(values) == 3
    assert abs(sum(value * value for value in values) - 1.0) < 1e-9
    # Same text in, same term ids out — this is what makes the query side match.
    assert build_sparse_terms("data integrity data integrity ALCOA")[0] == indices


def test_empty_text_yields_an_empty_sparse_vector():
    from app.infrastructure.embeddings import build_sparse_terms

    assert build_sparse_terms("   ") == ([], [])


# ── Registry v2 columns: artifact_id, content_role, is_preferred_source ───────
def test_artifact_id_is_the_unit_key_not_document_id(tmp_path, knowledge_base):
    """
    document_id identifies the logical document and may legitimately repeat —
    a guideline and its Q&A share one. artifact_id is unique per file, which is
    what state tracking and chunk ids need.
    """
    path = write_registry(tmp_path, [
        registry_row(document_id="ICH_Q7", artifact_id="ICH_Q7_GUIDE_AB12",
                     local_path="ich/q7/guide.pdf"),
    ])
    touch(tmp_path, "ich/q7/guide.pdf")

    entry = discover_documents(knowledge_base, path).documents[0]

    assert entry["document_key"] == "ICH_Q7_GUIDE_AB12"
    assert entry["document_id"] == "ICH_Q7"


def test_is_preferred_source_false_is_excluded_even_when_ingest_is_true(
    tmp_path, knowledge_base
):
    """
    Defence in depth against double-indexing: the registry marks alternate
    representations both ways, and either flag alone is enough to exclude.
    """
    path = write_registry(tmp_path, [
        registry_row(document_id="FDA_211", local_path="fda/211.json",
                     file_type="json", is_preferred_source="True"),
        registry_row(document_id="FDA_211", local_path="fda/211.md",
                     file_type="md", ingest="True", is_preferred_source="False"),
    ])
    touch(tmp_path, "fda/211.json")
    touch(tmp_path, "fda/211.md")

    discovery = discover_documents(knowledge_base, path)

    assert [entry["local_path"] for entry in discovery.documents] == ["fda/211.json"]
    assert len(discovery.not_ingested) == 1
    # It was excluded by the registry, so the format-preference fallback never
    # had to fire.
    assert not discovery.skipped_representations


def test_is_preferred_source_defaults_to_true_when_the_column_is_absent(tmp_path):
    """An older registry must not exclude every row for lacking the column."""
    path = tmp_path / "reg.csv"
    path.write_text(
        "document_id,document_title,local_path,ingest\nA,Title A,ich/a.pdf,True\n",
        encoding="utf-8-sig",
    )
    assert load_registry(path)[0]["is_preferred_source"] is True


def test_payload_carries_artifact_id_and_content_role():
    entry = dict(_entry(), artifact_id="ICH_Q9_PDF_1A2B", content_role="primary",
                 local_path="ich/q9/ICH_Q9.pdf", source_file="ICH_Q9.pdf")

    payload = kbi.build_payload(entry, _chunk(), 0, "id#0")

    assert payload["artifact_id"] == "ICH_Q9_PDF_1A2B"
    assert payload["content_role"] == "primary"


def test_fingerprint_tracks_content_role_and_document_id():
    """
    Both reach the payload, so both must invalidate stored chunks. The registry
    edit that split ICH_ICH_Q7 into ICH_ICH_Q7 / ICH_ICH_Q7_Q_A changed exactly
    these two fields and nothing else.
    """
    file_hash = "a" * 64
    baseline = kbi.document_fingerprint(_entry(), file_hash)

    assert kbi.document_fingerprint(
        _entry(content_role="q&a"), file_hash) != baseline
    assert kbi.document_fingerprint(
        _entry(document_id="ICH_Q9_Q_A"), file_hash) != baseline


def test_extraction_cache_is_keyed_by_file_hash_not_registry_key(tmp_path):
    """
    Renaming a document_id, or moving the unit key to artifact_id, must not
    throw away the extraction cache — re-reading the corpus costs ~50 minutes.
    """
    digest = "f" * 64
    first = extraction_cache.cache_path(tmp_path, digest)
    assert digest in first.name
    # Same file, whatever the registry now calls it -> same cache entry.
    assert extraction_cache.cache_path(tmp_path, digest) == first
