"""Claude API backend via the anthropic SDK."""

import os

from llm_pso.backends.base import BaseLLMBackend


class ClaudeBackend(BaseLLMBackend):
    """Claude API backend using the anthropic Python SDK."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
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
        return message.content[0].text

    def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
