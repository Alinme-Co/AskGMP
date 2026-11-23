"""
Tests for the document extractors and the upload service.

Fixtures are real files built in a tmp_path — a genuine .xlsx written by
openpyxl, a genuine .pptx written by python-pptx — because the things most
likely to break are the library APIs themselves: how openpyxl exposes an
anchored picture, how python-pptx reports a grouped shape. A mocked workbook
would test the mock.

OCR is stubbed. The engine is exercised for real elsewhere; here what matters
is that each reader finds the pictures, sends them to OCR, and turns the result
into a record with the right provenance — none of which needs a 40 MB ONNX
model in the loop.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest


from app.documents import models as R
from app.documents.extractors import (
    SUPPORTED_SUFFIXES,
    UPLOAD_TYPES,
    DocumentReadFailed,
    UnsupportedFormat,
    format_label,
    is_supported,
    read_document,
)
from app.uploads.service import process_upload, process_uploads
from app.uploads.storage import (
    UploadRejected,
    content_hash,
    safe_name,
    store,
)


# ── Stub OCR ──────────────────────────────────────────────────────────────────
OCR_TEXT = "OOS RESULT CONFIRMED\nPhase II investigation required"


class StubOCR:
    """Stands in for the RapidOCR engine; records what it was asked to read."""

    def __init__(self, text: str = OCR_TEXT, available: bool = True):
        self.text = text
        self._available = available
        self.calls: list[int] = []

    @property
    def available(self) -> bool:
        return self._available

    def text_from_bytes(self, data: bytes) -> str:
        self.calls.append(len(data))
        return self.text


@pytest.fixture
def ocr():
    return StubOCR()


@pytest.fixture
def png() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 120), "white")
    ImageDraw.Draw(image).text((20, 40), "OOS RESULT CONFIRMED", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ── Fixture documents ─────────────────────────────────────────────────────────
@pytest.fixture
def xlsx(tmp_path: Path, png: bytes) -> Path:
    import openpyxl
    from openpyxl.drawing.image import Image as XlImage

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Specs"
    sheet.append(["Parameter", "Limit", "Reference"])
    sheet.append(["Storage temperature", "2-8 C", "SOP-114"])
    sheet.append(["pH", "6.8-7.2", "SOP-115"])
    sheet.add_image(XlImage(io.BytesIO(png)), "E2")

    other = workbook.create_sheet("Deviations")
    other.append(["ID", "Description"])
    other.append(["DEV-001", "Temperature excursion"])

    hidden = workbook.create_sheet("Scratch")
    hidden.append(["ignore", "me"])
    hidden.sheet_state = "hidden"

    path = tmp_path / "spec.xlsx"
    workbook.save(path)
    return path


@pytest.fixture
def docx_file(tmp_path: Path, png: bytes) -> Path:
    import docx

    document = docx.Document()
    document.add_heading("Deviation Management", level=1)
    document.add_paragraph("Deviations shall be recorded and investigated.")
    document.add_heading("Investigation", level=2)
    document.add_paragraph("Root cause analysis within 30 days.")

    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Severity"
    table.rows[0].cells[1].text = "Timeline"
    table.rows[1].cells[0].text = "Critical"
    table.rows[1].cells[1].text = "24 hours"

    document.sections[0].footer.paragraphs[0].text = "SOP-114 Rev 3"
    document.add_picture(io.BytesIO(png))

    path = tmp_path / "sop.docx"
    document.save(path)
    return path


@pytest.fixture
def pptx_file(tmp_path: Path, png: bytes) -> Path:
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Out of Specification"

    box = slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(4), Inches(1))
    box.text_frame.text = "Phase I laboratory investigation"

    table = slide.shapes.add_table(2, 2, Inches(0.5), Inches(3), Inches(5), Inches(1)).table
    table.cell(0, 0).text = "Phase"
    table.cell(0, 1).text = "Action"
    table.cell(1, 0).text = "II"
    table.cell(1, 1).text = "Full investigation"

    slide.shapes.add_picture(io.BytesIO(png), Inches(5.5), Inches(1.6), width=Inches(3))
    slide.notes_slide.notes_text_frame.text = "Remember the retest rules."

    path = tmp_path / "oos.pptx"
    presentation.save(path)
    return path


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# ── models.py ─────────────────────────────────────────────────────────────────
def test_normalise_collapses_whitespace_not_content():
    assert R.normalise("a  b\r\n\r\n\r\n\r\nc  ") == "a b\n\nc"


def test_record_carries_kind_and_locator_into_meta():
    item = R.record("text", kind=R.KIND_CELL, locator="Sheet 'S'!B2", sheet="S", row=2)
    assert item["meta"]["kind"] == "cell"
    assert item["meta"]["locator"] == "Sheet 'S'!B2"
    assert item["meta"]["sheet"] == "S"
    assert item["section"] == "Sheet 'S'!B2"


def test_record_omits_provenance_that_is_none():
    item = R.record("text", kind=R.KIND_TEXT, sheet=None, row=4)
    assert "sheet" not in item["meta"]
    assert item["meta"]["row"] == 4


@pytest.mark.parametrize("call,expected", [
    (lambda: R.sheet_locator("Specs", cell="B2"), "Sheet 'Specs'!B2"),
    (lambda: R.sheet_locator("Specs", row=7), "Sheet 'Specs', row 7"),
    (lambda: R.slide_locator(3, "Overview"), "Slide 3 — Overview"),
    (lambda: R.table_locator(2, slide=4), "Table 2 (slide 4)"),
    (lambda: R.figure_locator(1, page=9), "Figure 1 on page 9"),
])
def test_locators_read_the_same_way_everywhere(call, expected):
    assert call() == expected


def test_summarise_counts_kinds_and_ocr():
    summary = R.summarise([
        R.record("a", kind=R.KIND_ROW),
        R.record("b", kind=R.KIND_ROW),
        R.record("c", kind=R.KIND_IMAGE, ocr=True, page=4),
    ])
    assert summary["records"] == 3
    assert summary["kinds"]["row"] == 2
    assert summary["ocr_records"] == 1
    assert summary["pages"] == (4, 4)


# ── Excel ─────────────────────────────────────────────────────────────────────
def test_excel_renders_rows_with_their_column_headers(xlsx, ocr):
    items = read_document(xlsx, ocr=ocr)
    rows = [item for item in items if item["meta"]["kind"] == "row"]

    assert any("Parameter: Storage temperature" in item["text"]
               and "Limit: 2-8 C" in item["text"] for item in rows), \
        "a value must never be indexed without the column that names it"


def test_excel_records_carry_sheet_and_row_provenance(xlsx, ocr):
    rows = [i for i in read_document(xlsx, ocr=ocr) if i["meta"]["kind"] == "row"]
    first = rows[0]
    assert first["meta"]["sheet"] == "Specs"
    assert first["meta"]["row"] == 2
    assert first["meta"]["locator"] == "Sheet 'Specs', row 2"
    assert any(cell["ref"] == "A2" for cell in first["meta"]["cells"])


def test_excel_reads_every_visible_sheet(xlsx, ocr):
    sheets = {i["meta"].get("sheet") for i in read_document(xlsx, ocr=ocr)}
    assert {"Specs", "Deviations"} <= sheets


def test_excel_skips_hidden_sheets(xlsx, ocr):
    """A hidden sheet is scratch working, not published content."""
    sheets = {i["meta"].get("sheet") for i in read_document(xlsx, ocr=ocr)}
    assert "Scratch" not in sheets


def test_excel_ocrs_anchored_pictures(xlsx, ocr):
    images = [i for i in read_document(xlsx, ocr=ocr) if i["meta"]["kind"] == "image"]
    assert len(images) == 1
    assert images[0]["meta"]["ocr"] is True
    assert images[0]["meta"]["sheet"] == "Specs"
    assert OCR_TEXT.splitlines()[0] in images[0]["text"]
    assert ocr.calls, "the picture must actually reach the OCR engine"


def test_excel_skips_ocr_when_the_engine_is_unavailable(xlsx):
    items = read_document(xlsx, ocr=StubOCR(available=False))
    assert not [i for i in items if i["meta"]["kind"] == "image"]


def test_excel_rows_are_atomic(xlsx, ocr):
    rows = [i for i in read_document(xlsx, ocr=ocr) if i["meta"]["kind"] == "row"]
    assert all(item["atomic"] for item in rows), \
        "gluing two unrelated rows makes a chunk that belongs to neither"


def test_excel_resolves_merged_cells(tmp_path, ocr):
    """openpyxl reports a merged value only on the top-left cell."""
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "M"
    sheet["A1"] = "Batch Record"
    sheet.merge_cells("A1:C1")
    sheet.append(["Step", "Result", "By"])
    sheet.append(["Weigh", "OK", "AB"])
    path = tmp_path / "merged.xlsx"
    workbook.save(path)

    text = " ".join(i["text"] for i in read_document(path, ocr=ocr))
    assert "Batch Record" in text


# ── Word ──────────────────────────────────────────────────────────────────────
def test_word_keeps_headings_as_the_section_of_following_prose(docx_file, ocr):
    items = read_document(docx_file, ocr=ocr)
    body = [i for i in items if i["meta"]["kind"] == "paragraph"]

    assert body[0]["section"] == "Deviation Management"
    assert body[1]["section"] == "Investigation"


def test_word_reads_tables_as_atomic_records(docx_file, ocr):
    tables = [i for i in read_document(docx_file, ocr=ocr) if i["meta"]["kind"] == "table"]
    assert len(tables) == 1
    assert tables[0]["atomic"] is True
    assert "Severity | Timeline" in tables[0]["text"]
    assert "Critical | 24 hours" in tables[0]["text"]


def test_word_keeps_body_order_so_a_table_stays_with_its_paragraph(docx_file, ocr):
    """
    python-docx exposes paragraphs and tables as two lists; reading them in
    turn would push every table to the end of the document.
    """
    kinds = [i["meta"]["kind"] for i in read_document(docx_file, ocr=ocr)]
    assert kinds.index("table") > kinds.index("paragraph")
    assert kinds.index("table") < len(kinds) - 1  # not simply appended last


def test_word_reads_the_footer(docx_file, ocr):
    """The footer is where the SOP number and revision live."""
    headers = [i for i in read_document(docx_file, ocr=ocr) if i["meta"]["kind"] == "header"]
    assert any("SOP-114 Rev 3" in item["text"] for item in headers)


def test_word_ocrs_embedded_pictures(docx_file, ocr):
    images = [i for i in read_document(docx_file, ocr=ocr) if i["meta"]["kind"] == "image"]
    assert len(images) == 1
    assert images[0]["meta"]["ocr"] is True
    assert images[0]["meta"]["locator"] == "Figure 1"


def test_word_prose_is_mergeable(docx_file, ocr):
    body = [i for i in read_document(docx_file, ocr=ocr) if i["meta"]["kind"] == "paragraph"]
    assert not any(item["atomic"] for item in body)


# ── PowerPoint ────────────────────────────────────────────────────────────────
def test_powerpoint_makes_one_record_per_slide(pptx_file, ocr):
    slides = [i for i in read_document(pptx_file, ocr=ocr) if i["meta"]["kind"] == "slide"]
    assert len(slides) == 1
    assert slides[0]["page"] == 1
    assert slides[0]["atomic"] is True
    assert "Out of Specification" in slides[0]["text"]
    assert "Phase I laboratory investigation" in slides[0]["text"]


def test_powerpoint_keeps_speaker_notes(pptx_file, ocr):
    notes = [i for i in read_document(pptx_file, ocr=ocr) if i["meta"]["kind"] == "notes"]
    assert notes and "retest rules" in notes[0]["text"]


def test_powerpoint_reads_tables_separately(pptx_file, ocr):
    tables = [i for i in read_document(pptx_file, ocr=ocr) if i["meta"]["kind"] == "table"]
    assert len(tables) == 1
    assert "Phase | Action" in tables[0]["text"]
    assert tables[0]["meta"]["slide"] == 1


def test_powerpoint_ocrs_pictures(pptx_file, ocr):
    images = [i for i in read_document(pptx_file, ocr=ocr) if i["meta"]["kind"] == "image"]
    assert len(images) == 1
    assert images[0]["meta"]["slide"] == 1
    assert images[0]["meta"]["ocr"] is True


def test_powerpoint_descends_into_grouped_shapes(tmp_path, ocr):
    """
    A group has no text frame of its own. Checking only has_text_frame at the
    top level loses every label inside a grouped diagram — which in a GMP deck
    is usually the whole process flow.
    """
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    group = slide.shapes.add_group_shape()
    for index, label in enumerate(("Sampling step", "Testing step")):
        box = group.shapes.add_textbox(
            Inches(1 + index), Inches(1), Inches(2), Inches(0.6)
        )
        box.text_frame.text = label

    path = tmp_path / "grouped.pptx"
    presentation.save(path)

    text = " ".join(i["text"] for i in read_document(path, ocr=ocr))
    assert "Sampling step" in text
    assert "Testing step" in text


# ── HTML and XLIFF ────────────────────────────────────────────────────────────
def test_html_sections_on_headings_and_drops_navigation(tmp_path):
    path = write(tmp_path, "guide.html",
                 "<html><body><nav>Home Menu Search</nav><main>"
                 "<h1>Data Integrity</h1><p>Records must be attributable.</p>"
                 "<h2>ALCOA</h2><p>Accurate and contemporaneous.</p>"
                 "</main></body></html>")

    items = read_document(path)
    sections = [item["section"] for item in items]

    assert "Data Integrity" in sections
    assert "ALCOA" in sections
    assert "Home Menu Search" not in " ".join(i["text"] for i in items)


def test_xliff_makes_one_atomic_record_per_segment(tmp_path):
    path = write(tmp_path, "seg.xlf",
                 '<?xml version="1.0"?><xliff version="1.2">'
                 '<file source-language="en" target-language="de"><body>'
                 '<trans-unit id="7"><source>Deviations shall be investigated.</source>'
                 '<target>Abweichungen sind zu untersuchen.</target></trans-unit>'
                 '<trans-unit id="8"><source>CAPA must be documented.</source></trans-unit>'
                 '</body></file></xliff>')

    items = read_document(path)
    assert len(items) == 2
    assert all(item["atomic"] for item in items)
    assert items[0]["meta"]["segment"] == "7"
    assert items[0]["meta"]["locator"] == "Segment 7"
    assert items[0]["text"] == "Deviations shall be investigated."


def test_xliff_keeps_the_translation_beside_the_source_not_inside_it(tmp_path):
    """Concatenating a target onto its source embeds two languages as one string."""
    path = write(tmp_path, "seg.xlf",
                 '<?xml version="1.0"?><xliff version="1.2">'
                 '<file source-language="en" target-language="de"><body>'
                 '<trans-unit id="1"><source>Deviations.</source>'
                 '<target>Abweichungen.</target></trans-unit>'
                 '</body></file></xliff>')

    item = read_document(path)[0]
    assert "Abweichungen" not in item["text"]
    assert item["meta"]["target_text"] == "Abweichungen."
    assert item["meta"]["source_language"] == "en"


# ── Plain formats ─────────────────────────────────────────────────────────────
def test_csv_labels_each_value_with_its_column(tmp_path):
    path = write(tmp_path, "batches.csv",
                 "Batch,Result,Status\nB-101,OOS,Under investigation\n")
    items = read_document(path)

    assert len(items) == 1
    assert "Batch: B-101" in items[0]["text"]
    assert "Status: Under investigation" in items[0]["text"]
    assert items[0]["meta"]["row"] == 2


def test_csv_without_headers_keeps_every_line(tmp_path):
    path = write(tmp_path, "numbers.csv", "1,2,3\n4,5,6\n")
    assert len(read_document(path)) == 2


def test_markdown_sections_on_headings(tmp_path):
    path = write(tmp_path, "notes.md", "# CAPA\n\nCorrective.\n\n## Preventive\n\nPreventive.\n")
    sections = [item["section"] for item in read_document(path)]
    assert "CAPA" in sections and "Preventive" in sections


def test_json_qa_becomes_one_record_per_pair(tmp_path):
    path = write(tmp_path, "qa.json", json.dumps([
        {"section": "DI", "question": "What is ALCOA?", "answer": "Attributable."},
        {"section": "DI", "question": "And ALCOA+?", "answer": "Complete."},
    ]))
    items = read_document(path)
    assert len(items) == 2
    assert all(item["atomic"] for item in items)


def test_plain_text_is_read(tmp_path):
    path = write(tmp_path, "plain.txt", "Change control requires impact assessment.")
    assert "impact assessment" in read_document(path)[0]["text"]


# ── Dispatch ──────────────────────────────────────────────────────────────────
def test_every_promised_format_is_registered():
    for suffix in (".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm",
                   ".xliff", ".xlf", ".md", ".json", ".csv", ".txt"):
        assert suffix in SUPPORTED_SUFFIXES, f"{suffix} has no reader"


def test_upload_types_have_no_leading_dot():
    """st.file_uploader wants extensions without the dot."""
    assert all(not item.startswith(".") for item in UPLOAD_TYPES)


def test_unknown_extension_is_refused_clearly(tmp_path):
    path = write(tmp_path, "thing.exe", "MZ")
    with pytest.raises(UnsupportedFormat):
        read_document(path)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(DocumentReadFailed):
        read_document(tmp_path / "nope.docx")


def test_a_corrupt_file_raises_rather_than_returning_nothing(tmp_path):
    """An empty result would look like a blank document instead of a failure."""
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a docx")
    with pytest.raises(DocumentReadFailed):
        read_document(path)


def test_format_label_and_is_supported():
    assert format_label(".xlsx") == "Excel"
    assert format_label(".pptx") == "PowerPoint"
    assert is_supported("a/b/c.pdf")
    assert not is_supported("a/b/c.exe")


# ── Uploads ───────────────────────────────────────────────────────────────────
def test_upload_stores_reads_and_summarises(xlsx, tmp_path):
    document = process_upload(
        xlsx.read_bytes(), "spec.xlsx", use_ocr=False, upload_dir=tmp_path / "up"
    )
    assert document.ok
    assert document.format_label == "Excel"
    assert document.summary["records"] == len(document.records)
    assert document.path.exists()
    assert document.text


def test_upload_filename_cannot_escape_the_upload_directory():
    assert "/" not in safe_name("../../.env")
    assert "\\" not in safe_name("..\\..\\.env")
    assert safe_name("../../.env") == "env"


def test_upload_name_is_sanitised_but_recognisable():
    assert safe_name("Batch Record (final) v2.xlsx") == "Batch_Record_final_v2.xlsx"


def test_upload_rejects_an_unsupported_extension(tmp_path):
    with pytest.raises(UnsupportedFormat):
        store(b"MZ", "virus.exe", upload_dir=tmp_path)


def test_upload_rejects_an_empty_file(tmp_path):
    with pytest.raises(UploadRejected):
        store(b"", "empty.txt", upload_dir=tmp_path)


def test_upload_rejects_an_oversized_file(tmp_path, monkeypatch):
    from app.uploads import storage

    monkeypatch.setattr(storage, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(UploadRejected):
        storage.store(b"x" * 50, "big.txt", upload_dir=tmp_path)


def test_the_same_bytes_store_under_the_same_name(tmp_path):
    first, digest_a, _ = store(b"hello", "notes.txt", upload_dir=tmp_path)
    second, digest_b, _ = store(b"hello", "notes.txt", upload_dir=tmp_path)
    assert first == second and digest_a == digest_b
    assert len(list(tmp_path.iterdir())) == 1


def test_content_hash_identifies_a_file_before_it_is_read():
    assert content_hash(b"a") == content_hash(b"a")
    assert content_hash(b"a") != content_hash(b"b")


def test_a_corrupt_upload_reports_instead_of_raising(tmp_path):
    document = process_upload(b"not a docx", "broken.docx", upload_dir=tmp_path)
    assert not document.ok
    assert "Word" in document.error


def test_one_bad_file_does_not_lose_the_good_ones(tmp_path, xlsx):
    class Fake:
        def __init__(self, name, data):
            self.name, self._data = name, data

        def getvalue(self):
            return self._data

    results = process_uploads(
        [Fake("spec.xlsx", xlsx.read_bytes()), Fake("broken.docx", b"nope")],
        use_ocr=False,
        upload_dir=tmp_path,
    )
    assert len(results) == 2
    assert results[0].ok and not results[1].ok


def test_upload_dict_is_json_safe(xlsx, tmp_path):
    document = process_upload(
        xlsx.read_bytes(), "spec.xlsx", use_ocr=False, upload_dir=tmp_path
    )
    json.dumps(document.to_dict())      # must not raise
