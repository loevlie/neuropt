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


DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend(BaseLLMBackend):
    """OpenAI chat completions, or any server that speaks the same API.

    Args:
        model: Model id. Defaults to ``$NEUROPT_MODEL``, else ``gpt-4o-mini``.
        base_url: API root for an OpenAI-compatible server (vLLM, Ollama,
            llama.cpp, LM Studio, OpenRouter). Defaults to
            ``$NEUROPT_LLM_BASE_URL``, else OpenAI itself.
        api_key: Defaults to ``$NEUROPT_LLM_API_KEY`` for a custom
            ``base_url``, else ``$OPENAI_API_KEY``.

    The answer is constrained to the caller's JSON schema (``response_format``
    with ``strict``) when one is given; servers that reject it get plain JSON
    instructions instead. Cost is only reported for OpenAI models in the price
    table; cached prompt tokens are priced as normal input, so it is an upper
    bound when OpenAI's automatic prompt caching applies.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None):
        super().__init__()
        # None: take $NEUROPT_LLM_BASE_URL; "" forces OpenAI itself
        if base_url is None:
            base_url = os.environ.get("NEUROPT_LLM_BASE_URL")
        self.base_url = base_url or None
        env_model = os.environ.get("NEUROPT_MODEL", "")
        if env_model.startswith("claude"):
            env_model = ""
        self._model = model or env_model or (None if self.base_url else DEFAULT_MODEL)
        if self._model is None:
            raise ValueError("an OpenAI-compatible base_url needs a model name "
                             "(model=... or $NEUROPT_MODEL)")
        if api_key is None:
            api_key = (os.environ.get("NEUROPT_LLM_API_KEY") if self.base_url
                       else os.environ.get("OPENAI_API_KEY"))
        self._api_key = api_key
        self._client = None
        self._structured = True

    @property
    def name(self) -> str:
        where = f" @ {self.base_url}" if self.base_url else ""
        return f"OpenAIBackend({self._model}{where})"

    def _get_client(self):
        if self._client is None:
            import openai
            # Local servers ignore the key, but the SDK insists on one
            key = self._api_key or ("not-needed" if self.base_url else None)
            self._client = openai.OpenAI(base_url=self.base_url, api_key=key)
        return self._client

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.generate_structured(prompt, max_tokens=max_tokens)

    def generate_structured(self, prompt, system=None, schema=None, max_tokens=1024):
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        # max_completion_tokens works for every chat model; reasoning models
        # (o-series, gpt-5) reject the older max_tokens parameter.
        kwargs = dict(model=self._model, max_completion_tokens=max_tokens, messages=messages)
        if schema is not None and self._structured:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "configs", "schema": schema, "strict": True}}
        try:
            response = self._get_client().chat.completions.create(**kwargs)
        except Exception as e:
            if "response_format" not in kwargs or getattr(e, "status_code", None) not in (400, 422):
                raise
            print(f"  [{self.name} rejected response_format ({str(e)[:120]}); "
                  "continuing with plain JSON instructions]")
            self._structured = False
            del kwargs["response_format"]
            response = self._get_client().chat.completions.create(**kwargs)
        if response.usage is not None:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0
        return response.choices[0].message.content or ""

    def is_available(self) -> bool:
        return bool(self.base_url or self._api_key)

    @property
    def total_cost(self) -> float | None:
        if getattr(self, "base_url", None):
            return None  # unknown pricing (free for a local server)
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
