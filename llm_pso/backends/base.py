"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod


class BaseLLMBackend(ABC):
    """Base class for LLM backends used by the advisor."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate a text response from the given prompt.

        Args:
            prompt: The full prompt text.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text response.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is ready to use.

        Returns:
            True if the backend can generate responses.
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__
