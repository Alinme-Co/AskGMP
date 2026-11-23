"""
ocr.py — one OCR engine, shared by every extractor.

Wraps `app/documents/pdf_engine.py`'s DocumentReader rather than
reimplementing it. That module already solves the parts that are easy to get
subtly wrong — finding the right det/rec/cls ONNX model in OCR/, matching the
PP-OCR version a model expects, and turning RapidOCR's box output back into
reading order rather than detection order — and it is proven against this
project's own model files. Copying that logic to serve pictures out of a
workbook would create a second thing to keep correct.

What this adds is the entry point the non-PDF extractors need. pdf_engine renders
PDF pages to numpy arrays itself; a picture pulled out of a .docx or .pptx
arrives as PNG/JPEG bytes, so `text_from_bytes` is the bridge.

The engine is built once and lazily. Loading three ONNX models costs a second
or two, and a document with no pictures in it should never pay that.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Pictures below this, in pixels, are bullets, rules and logos rather than
# anything with readable text in them.
MIN_IMAGE_PIXELS = 40


class ImageOCR:
    """
    Text out of an image, or nothing.

    Every failure — RapidOCR missing, a model file absent, an unreadable
    picture — degrades to an empty string rather than raising. A reader's job
    is to get what text it can out of a document; one unreadable diagram must
    not lose the other forty pages.
    """

    def __init__(self, enabled: bool = True, text_score: float = 0.5) -> None:
        self.enabled = enabled
        self.text_score = text_score
        self._reader: Any = None
        self._unavailable_reason: str | None = None

    # -- engine --
    @property
    def reader(self):
        """The underlying DocumentReader, built on first use."""
        if self._reader is None and self._unavailable_reason is None:
            try:
                from app.documents.pdf_engine import DocumentReader

                self._reader = DocumentReader(text_score=self.text_score)
                # Touch the engine now so a missing model surfaces here rather
                # than once per picture.
                _ = self._reader.ocr_engine
            except Exception as exc:              # noqa: BLE001
                self._unavailable_reason = f"{type(exc).__name__}: {exc}"
                logger.warning("OCR unavailable (%s)", self._unavailable_reason)
        return self._reader

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        return self.reader is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    # -- use --
    def text_from_bytes(self, data: bytes) -> str:
        """
        OCR one embedded picture.

        Decoded through Pillow because .docx, .xlsx and .pptx store pictures as
        whatever the author pasted in — PNG, JPEG, EMF, WMF. Pillow reads the
        common ones and refuses the vector formats, which is the right outcome:
        an EMF chart has no raster text to find.
        """
        if not data or not self.available:
            return ""

        try:
            import io

            import numpy as np
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                if image.width < MIN_IMAGE_PIXELS or image.height < MIN_IMAGE_PIXELS:
                    return ""
                rgb = image.convert("RGB")
                array = np.asarray(rgb)
        except Exception as exc:                  # noqa: BLE001
            logger.debug("Could not decode an embedded image: %s", exc)
            return ""

        # RapidOCR reads a 3-channel array as BGR.
        import numpy as np

        bgr = np.ascontiguousarray(array[:, :, ::-1])
        try:
            return (self.reader.ocr_image(bgr) or "").strip()
        except Exception as exc:                  # noqa: BLE001
            logger.debug("OCR failed on an embedded image: %s", exc)
            return ""


_SHARED: ImageOCR | None = None


def shared_ocr(enabled: bool = True) -> ImageOCR:
    """
    One engine per process.

    The ONNX session is the expensive part, and a request that reads a
    forty-picture deck should build it once.
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = ImageOCR(enabled=enabled)
    _SHARED.enabled = enabled
    return _SHARED
