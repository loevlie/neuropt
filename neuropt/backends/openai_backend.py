"""OpenAI API backend via the openai SDK."""

import os
from neuropt.backends.base import BaseLLMBackend


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
        return response.choices[0].message.content

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))
