"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod


class BaseLLMBackend(ABC):

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate a text response from the given prompt."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is ready to use."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def total_cost(self) -> float | None:
        """Return estimated cost in USD, or None if unknown."""
        return None
