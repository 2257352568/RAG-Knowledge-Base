import logging
from imageproc.ocr import BaseOCR
from imageproc.vlm import BaseVLM

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Orchestrate OCR + VLM on a single image, merge results.

    Usage:
        proc = ImageProcessor(ocr=TesseractOCR(), vlm=OpenAIVLM())
        desc = proc.process(image_bytes, context="Figure 1: Architecture overview")
    """

    def __init__(self, ocr: BaseOCR | None = None, vlm: BaseVLM | None = None):
        self._ocr = ocr
        self._vlm = vlm

    @property
    def enabled(self) -> bool:
        return self._ocr is not None or self._vlm is not None

    def process(self, image_bytes: bytes, context: str = "") -> str:
        """Process an image: OCR + VLM → merged description string."""
        if not self.enabled:
            return ""

        parts: list[str] = []

        if self._ocr:
            try:
                ocr_text = self._ocr.extract(image_bytes)
                if ocr_text:
                    parts.append(f"OCR文字: {ocr_text}")
            except Exception as e:
                logger.warning("OCR failed: %s", e)

        if self._vlm:
            try:
                vlm_desc = self._vlm.describe(image_bytes, context)
                if vlm_desc:
                    parts.append(f"视觉描述: {vlm_desc}")
            except Exception as e:
                logger.warning("VLM failed: %s", e)

        if not parts:
            return ""

        return "[图片内容] " + " | ".join(parts)
