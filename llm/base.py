from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseLLM(ABC):
    """Abstract LLM backend — swap between DeepSeek, OpenAI, local models, etc."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion from a list of messages."""
        ...

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Stream completion chunks. Override in subclass for real streaming."""
        yield self.generate(messages, temperature=temperature, max_tokens=max_tokens)
