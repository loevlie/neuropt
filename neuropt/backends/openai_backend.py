"""OpenAI API backend via the openai SDK."""

import os

from neuropt.backends.base import BaseLLMBackend

# USD per million tokens — https://platform.openai.com/docs/pricing
_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


class OpenAIBackend(BaseLLMBackend):

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__()
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.usage is not None:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    @property
    def total_cost(self) -> float | None:
        prices = _PRICING.get(self._model)
        if prices is None:
            # Longest prefix first so "gpt-4o-mini-2024-07-18" matches gpt-4o-mini, not gpt-4o
            for prefix, p in sorted(_PRICING.items(), key=lambda kv: -len(kv[0])):
                if self._model.startswith(prefix):
                    prices = p
                    break
        if prices is None:
            return None
        input_price, output_price = prices
        return (self.total_input_tokens * input_price +
                self.total_output_tokens * output_price) / 1_000_000
