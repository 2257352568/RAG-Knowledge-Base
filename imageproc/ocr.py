import io
from abc import ABC, abstractmethod


class BaseOCR(ABC):
    """Abstract OCR backend."""

    @abstractmethod
    def extract(self, image_bytes: bytes) -> str:
        """Extract text from image bytes. Returns empty string if no text found."""
        ...


class TesseractOCR(BaseOCR):
    """Tesseract OCR — industry standard, requires system install.

    Install: sudo apt install tesseract-ocr tesseract-ocr-chi-sim
    """

    def __init__(self, lang: str = "chi_sim+eng"):
        self._lang = lang

    def extract(self, image_bytes: bytes) -> str:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang=self._lang)
        return text.strip()


class EasyOCR(BaseOCR):
    """EasyOCR — pip-installable, supports 80+ languages including Chinese.

    First run downloads model files (~100MB). No system dependencies.
    """

    def __init__(self, langs: list[str] | None = None):
        self._langs = langs or ["ch_sim", "en"]
        self._reader = None

    def _init_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._langs)

    def extract(self, image_bytes: bytes) -> str:
        self._init_reader()
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        arr = np.array(img)
        results = self._reader.readtext(arr, detail=0)
        return "\n".join(results) if results else ""
