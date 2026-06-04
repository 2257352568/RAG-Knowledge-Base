import base64
import io
from abc import ABC, abstractmethod

from PIL import Image


class BaseVLM(ABC):
    """Abstract VLM backend — generate textual descriptions of images."""

    @abstractmethod
    def describe(self, image_bytes: bytes, context: str = "") -> str:
        """Generate a description of the image. Optional surrounding text context."""
        ...


class OpenAIVLM(BaseVLM):
    """OpenAI-compatible Vision API (GPT-4V, Claude Vision, etc.)."""

    def __init__(
        self,
        api_key: str = "sk-xxx",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
    ):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        img_b64 = base64.b64encode(image_bytes).decode()
        mime = self._guess_mime(image_bytes)

        prompt = "请详细描述这张图片的内容，包括图表类型、关键数据、以及图片传达的主要信息。用中文回答。"
        if context:
            prompt = f"这张图片出现在以下上下文中:\n{context}\n\n{prompt}"

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                ],
            }],
            max_tokens=500,
        )
        return resp.choices[0].message.content or ""

    def _guess_mime(self, image_bytes: bytes) -> str:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format.lower() if img.format else "png"
        return f"image/{fmt}"


class DeepSeekVLM(OpenAIVLM):
    """DeepSeek Vision API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str = "sk-xxx",
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model)


class DummyVLM(BaseVLM):
    """Offline placeholder — returns image metadata instead of semantic description."""

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return f"[图像: {img.format} | 尺寸: {img.size[0]}x{img.size[1]} | 模式: {img.mode}]"
        except Exception:
            return "[图像: 无法解析]"
