"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod


class BaseLLMBackend(ABC):

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate a text response from the given prompt."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is ready to use."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
