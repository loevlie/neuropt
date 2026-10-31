"""Claude API backend via the anthropic SDK."""

import os

from neuropt.backends.base import BaseLLMBackend

# USD per million tokens (input, output) — https://docs.anthropic.com/en/docs/about-claude/pricing
# Matched by longest prefix, so dated snapshots ("claude-haiku-4-5-20251001")
# resolve to their family entry. Unknown models report no cost rather than a guess.
_PRICING = {
    "claude-fable-5": (10.00, 50.00),      # Fable 5 and 5.1
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),      # prompts up to 100K tokens
    "claude-haiku-4-5": (1.00, 5.00),
}


class ClaudeBackend(BaseLLMBackend):
    """Claude via the Messages API.

    The default stays on Claude Haiku 4.5 because the published benchmarks
    were run with it; pass ``model=...`` to use another Claude model.
    """

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
        return "".join(block.text for block in message.content if block.type == "text")

    def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def total_cost(self) -> float | None:
        prices = None
        for prefix, p in sorted(_PRICING.items(), key=lambda kv: -len(kv[0])):
            if self._model.startswith(prefix):
                prices = p
                break
        if prices is None:
            return None
        input_price, output_price = prices
        return (self.total_input_tokens * input_price +
                self.total_output_tokens * output_price) / 1_000_000
