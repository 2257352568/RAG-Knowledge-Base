import os
from collections.abc import Iterator

from openai import OpenAI
from core.llm.base import BaseLLM


class DeepSeekLLM(BaseLLM):
    """DeepSeek API — OpenAI-compatible protocol.

    Models:
        - deepseek-chat      (general chat, 128K context)
        - deepseek-reasoner  (reasoning / R1, 128K context)

    Docs: https://platform.deepseek.com/api-docs
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        self._client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY", "sk-xxx"),
            base_url=base_url,
        )
        self._model = model

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
