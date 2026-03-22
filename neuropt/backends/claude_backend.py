"""Claude API backend via the anthropic SDK."""

import os
from neuropt.backends.base import BaseLLMBackend

# USD per million tokens — https://docs.anthropic.com/en/docs/about-claude/models
_PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-5-20250514": (3.00, 15.00),
    "claude-opus-4-6-20250624": (15.00, 75.00),
}
# Fallback: match by family name prefix
_FAMILY_PRICING = {
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
}


class ClaudeBackend(BaseLLMBackend):

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        super().__init__()
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self.total_input_tokens += message.usage.input_tokens
        self.total_output_tokens += message.usage.output_tokens
        return message.content[0].text

    def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def total_cost(self) -> float | None:
        prices = _PRICING.get(self._model)
        if prices is None:
            for prefix, p in _FAMILY_PRICING.items():
                if self._model.startswith(prefix):
                    prices = p
                    break
        if prices is None:
            return None
        input_price, output_price = prices
        return (self.total_input_tokens * input_price +
                self.total_output_tokens * output_price) / 1_000_000
