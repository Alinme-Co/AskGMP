"""
pdf_engine.py — PDF text extraction with OCR, and the shared OCR engine.

Combines two sources of text for a PDF:

* the native text layer (fast, exact), read with PyMuPDF, and
* OCR of the raster content — whole pages when a page turns out to be a scan,
  and the embedded pictures otherwise — read with RapidOCR using the ONNX
  models in the ``OCR/`` folder beside this file.

This is the engine, not a reader: it returns structured per-page results and
leaves the decision of what to do with them to the PDF extractor,
which splits page prose and figures into separate records so a citation can
point at a diagram rather than at the page it sat on.

`ocr_image` is also the entry point every *other* format uses. A picture pulled
out of a .docx or .pptx arrives as PNG bytes rather than a PDF page, and
app/documents/ocr.py bridges the two, so there is one OCR engine in the project
rather than one per format.

Adapted from a PDF-to-text tool in an earlier project. The paths that produced
a whole document as a single string, and its command-line front end, are gone —
AskGMP consumes the structured `PageContent` results directly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pymupdf

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Folder holding the RapidOCR ONNX models, next to this file.
DEFAULT_OCR_DIR = Path(__file__).resolve().parent / "OCR"


class DocumentReadError(Exception):
    """Raised when a document cannot be opened or read."""


@dataclass
class ImageContent:
    """OCR result for a single image region of a page."""

    index: int
    bbox: Tuple[float, float, float, float]
    text: str = ""

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


@dataclass
class PageContent:
    """Everything that was read from one PDF page."""

    number: int  # 1-based
    text: str = ""  # native text layer
    ocr_text: str = ""  # OCR of the full page (scanned pages only)
    images: List[ImageContent] = field(default_factory=list)
    is_scanned: bool = False


class DocumentReader:
    """Read PDFs into structured pages, OCR-ing whatever is not selectable text.

    Args:
        ocr_dir: Folder containing the RapidOCR ONNX models.  Defaults to the
            ``OCR/`` folder next to this module.
        use_ocr: Set to ``False`` to read only the native text layer.
        page_dpi: Render resolution used when a whole page has to be OCR-ed.
        image_dpi: Render resolution used for embedded image regions.  Higher
            than ``page_dpi`` because these regions are usually small.
        min_native_chars: A page whose native text layer has fewer than this
            many characters is treated as scanned and OCR-ed as a whole.
        min_image_size: Image regions smaller than this (in PDF points, i.e.
            1/72 inch) are considered decorative and skipped.
        text_score: Minimum RapidOCR confidence for a recognised line.
        sort_text: Order the native text layer by position instead of by
            content-stream order.
        max_render_pixels: Safety cap on the size of a single rendered bitmap;
            the effective DPI is reduced for pages that would exceed it.
    """

    def __init__(
        self,
        ocr_dir: Optional[PathLike] = None,
        use_ocr: bool = True,
        page_dpi: int = 200,
        image_dpi: int = 300,
        min_native_chars: int = 50,
        min_image_size: float = 40.0,
        text_score: float = 0.5,
        sort_text: bool = False,
        max_render_pixels: int = 40_000_000,
    ) -> None:
        self.ocr_dir = Path(ocr_dir) if ocr_dir else DEFAULT_OCR_DIR
        self.use_ocr = use_ocr
        self.page_dpi = page_dpi
        self.image_dpi = image_dpi
        self.min_native_chars = min_native_chars
        self.min_image_size = min_image_size
        self.text_score = text_score
        self.sort_text = sort_text
        self.max_render_pixels = max_render_pixels

        self._engine = None  # lazily built RapidOCR instance

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def read_pdf_pages(
        self,
        pdf_path: PathLike,
        pages: Optional[Sequence[int]] = None,
        password: Optional[str] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[PageContent]:
        """Extract every page of ``pdf_path``, including text inside images.

        Args:
            pdf_path: Path to the PDF file.
            pages: Optional 0-based page indices to limit the extraction to.
            password: Password for encrypted PDFs.
            progress: Optional ``callback(pages_done, total_pages)`` invoked
                after every page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise DocumentReadError(f"File not found: {pdf_path}")

        doc = self._open_pdf(pdf_path, password)
        try:
            indices = self._page_indices(doc, pages)
            total = len(indices)
            results: List[PageContent] = []
            for position, page_index in enumerate(indices, start=1):
                results.append(self._read_page(doc, page_index))
                if progress is not None:
                    progress(position, total)
            return results
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Per-page work
    # ------------------------------------------------------------------
    def _read_page(self, doc: "pymupdf.Document", page_index: int) -> PageContent:
        page = doc[page_index]
        content = PageContent(number=page_index + 1)

        content.text = self.extract_text(page)
        content.is_scanned = self.is_scanned_page(content.text)

        if not self.use_ocr:
            return content

        if content.is_scanned:
            # Nothing selectable on this page: OCR the whole rendered page.
            content.ocr_text = self.ocr_page(page)
        else:
            # Selectable text is present, so only the embedded images can hold
            # text we have not seen yet.
            content.images = self.extract_images(page)

        return content

    def extract_text(self, page: "pymupdf.Page") -> str:
        """Return the native (selectable) text layer of a page."""
        try:
            return page.get_text("text", sort=self.sort_text) or ""
        except Exception as exc:  # malformed page
            logger.warning("Could not read text of page %s: %s", page.number + 1, exc)
            return ""

    def is_scanned_page(self, text: str) -> bool:
        """True when a page carries too little real text to be a text page."""
        return len(re.sub(r"\s", "", text)) < self.min_native_chars

    def ocr_page(self, page: "pymupdf.Page") -> str:
        """OCR a full page by rendering it to a bitmap."""
        image = self.render(page, dpi=self.page_dpi)
        if image is None:
            return ""
        return self.ocr_image(image)

    def extract_images(self, page: "pymupdf.Page") -> List[ImageContent]:
        """OCR every meaningful image region on a page."""
        results: List[ImageContent] = []
        for index, rect in enumerate(self.image_rects(page), start=1):
            image = self.render(page, dpi=self.image_dpi, clip=rect)
            if image is None:
                continue
            text = self.ocr_image(image)
            if text.strip():
                results.append(ImageContent(index=index, bbox=tuple(rect), text=text))
        return results

    def image_rects(self, page: "pymupdf.Page") -> List["pymupdf.Rect"]:
        """Bounding boxes of the images on a page worth sending to OCR.

        Tiny decorations are dropped and boxes already covered by a larger box
        are skipped, so the same artwork is not OCR-ed twice.
        """
        try:
            infos = page.get_image_info()
        except Exception as exc:  # malformed page
            logger.warning("Could not list images on page %s: %s", page.number + 1, exc)
            return []

        page_rect = page.rect
        candidates: List["pymupdf.Rect"] = []
        for info in infos:
            rect = pymupdf.Rect(info["bbox"]) & page_rect
            if rect.is_empty or rect.is_infinite:
                continue
            if rect.width < self.min_image_size or rect.height < self.min_image_size:
                continue
            candidates.append(rect)

        # Largest first, then drop boxes contained in one we already kept.
        candidates.sort(key=lambda r: r.get_area(), reverse=True)
        kept: List["pymupdf.Rect"] = []
        for rect in candidates:
            covered = any(
                (rect & other).get_area() > 0.9 * rect.get_area() for other in kept
            )
            if not covered:
                kept.append(rect)

        # Restore top-to-bottom, left-to-right reading order.
        kept.sort(key=lambda r: (round(r.y0, 1), round(r.x0, 1)))
        return kept

    # ------------------------------------------------------------------
    # Rendering and OCR helpers
    # ------------------------------------------------------------------
    def render(
        self,
        page: "pymupdf.Page",
        dpi: int,
        clip: Optional["pymupdf.Rect"] = None,
    ) -> Optional[np.ndarray]:
        """Render a page (or a region of it) to a BGR numpy array."""
        rect = clip if clip is not None else page.rect
        dpi = self._fit_dpi(rect, dpi)
        zoom = dpi / 72.0
        try:
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom),
                clip=clip,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
        except Exception as exc:
            logger.warning("Could not render page %s: %s", page.number + 1, exc)
            return None

        if pix.width == 0 or pix.height == 0:
            return None

        rgb = np.frombuffer(pix.samples, dtype=np.uint8)
        rgb = rgb.reshape(pix.height, pix.width, 3)
        # RapidOCR treats a 3-channel array as BGR, so flip the channel order.
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def _fit_dpi(self, rect: "pymupdf.Rect", dpi: int) -> int:
        """Lower the DPI if the requested render would be too large."""
        pixels = (rect.width * dpi / 72.0) * (rect.height * dpi / 72.0)
        if pixels <= self.max_render_pixels or pixels <= 0:
            return dpi
        scale = (self.max_render_pixels / pixels) ** 0.5
        return max(72, int(dpi * scale))

    def ocr_image(self, image: np.ndarray) -> str:
        """Run RapidOCR on a BGR image and return its text, one line per row."""
        if image is None or image.size == 0:
            return ""
        try:
            result = self.ocr_engine(image)
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
            return ""
        return self._result_to_text(result)

    @staticmethod
    def _result_to_text(result) -> str:
        """Turn a RapidOCR result into text, ordered top-to-bottom."""
        txts = getattr(result, "txts", None)
        if not txts:
            return ""

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) != len(txts):
            return "\n".join(t for t in txts if t)

        # (top, left, bottom, text) for every detected box.
        items = []
        for text, box in zip(txts, boxes):
            box = np.asarray(box, dtype=float)
            items.append(
                (
                    float(box[:, 1].min()),
                    float(box[:, 0].min()),
                    float(box[:, 1].max()),
                    text,
                )
            )
        items.sort(key=lambda item: (item[0], item[1]))

        # Group into lines: a new line starts when a box sits clearly below the
        # bottom of the line being built.
        lines: List[List[Tuple[float, str]]] = []
        current: List[Tuple[float, str]] = []
        current_bottom = 0.0
        for top, left, bottom, text in items:
            height = max(bottom - top, 1.0)
            if current and top > current_bottom - 0.5 * height:
                lines.append(current)
                current = []
            if current:
                current_bottom = max(current_bottom, bottom)
            else:
                current_bottom = bottom
            current.append((left, text))
        if current:
            lines.append(current)

        rendered = []
        for line in lines:
            line.sort(key=lambda item: item[0])
            joined = " ".join(text for _, text in line if text).strip()
            if joined:
                rendered.append(joined)
        return "\n".join(rendered)

    # ------------------------------------------------------------------
    # OCR engine setup
    # ------------------------------------------------------------------
    @property
    def ocr_engine(self):
        """The RapidOCR engine, built on first use."""
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def _build_engine(self):
        try:
            from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
        except ImportError as exc:
            raise DocumentReadError(
                "RapidOCR is not installed. Run: pip install rapidocr onnxruntime"
            ) from exc

        det = self.find_model("det")
        rec = self.find_model("rec")
        cls = self.find_model("cls")

        params = {
            "Global.log_level": "error",
            "Global.text_score": self.text_score,
            "Global.use_cls": cls is not None,
        }
        for section, model_path in (("Det", det), ("Rec", rec), ("Cls", cls)):
            if model_path is None:
                continue
            params[f"{section}.engine_type"] = EngineType.ONNXRUNTIME
            params[f"{section}.model_type"] = ModelType.MOBILE
            params[f"{section}.model_path"] = str(model_path)
            params[f"{section}.ocr_version"] = self._model_version(model_path, OCRVersion)

        logger.debug("Building RapidOCR engine with %s", params)
        return RapidOCR(params=params)

    def find_model(self, task: str) -> Optional[Path]:
        """Locate the ONNX model for ``task`` (``det``, ``rec`` or ``cls``)."""
        if not self.ocr_dir.is_dir():
            raise DocumentReadError(f"OCR model folder not found: {self.ocr_dir}")

        needle = f"_{task}_"
        matches = [
            path
            for path in sorted(self.ocr_dir.glob("*.onnx"))
            if needle in path.name or path.stem.endswith(f"_{task}")
        ]
        if not matches:
            if task == "cls":
                return None  # the orientation classifier is optional
            raise DocumentReadError(f"No {task} ONNX model found in {self.ocr_dir}")
        return matches[0]

    @staticmethod
    def _model_version(model_path: Path, ocr_version_enum):
        """Guess the PP-OCR version a model belongs to from its file name."""
        match = re.search(r"PP-OCRv(\d)", model_path.name, re.IGNORECASE)
        if match:
            try:
                return ocr_version_enum(f"PP-OCRv{match.group(1)}")
            except ValueError:
                pass
        # The PP-LCNet textline orientation classifier ships with PP-OCRv5 and
        # expects that version's 80x160 input shape.
        return ocr_version_enum("PP-OCRv5")

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _open_pdf(pdf_path: Path, password: Optional[str]) -> "pymupdf.Document":
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as exc:
            raise DocumentReadError(f"Could not open {pdf_path}: {exc}") from exc

        if doc.needs_pass:
            if not password or not doc.authenticate(password):
                doc.close()
                raise DocumentReadError(f"{pdf_path} is password protected.")
        return doc

    @staticmethod
    def _page_indices(
        doc: "pymupdf.Document", pages: Optional[Sequence[int]]
    ) -> List[int]:
        if pages is None:
            return list(range(doc.page_count))
        indices = [p for p in pages if 0 <= p < doc.page_count]
        if not indices:
            raise DocumentReadError(
                f"No valid pages in {list(pages)} for a "
                f"{doc.page_count}-page document."
            )
        return indices
