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


DEFAULT_MODEL = "claude-haiku-5-5"


class ClaudeBackend(BaseLLMBackend):
    """Claude via the Messages API.

    The static part of the prompt is sent as a cached system prompt, and the
    answer is constrained to the caller's JSON schema (structured outputs)
    when one is given. Caching only takes effect above the model's minimum
    prompt length (4096 tokens on Haiku 4.5, 512 on the 5.5 models), so for
    short prompts it is a no-op, not an extra cost.

    Args:
        model: Model id. Defaults to ``$NEUROPT_MODEL`` if it names a Claude
            model, else Claude Haiku 5.5 (chosen on the benchmark harness: never
            worse than Haiku 4.5 by the pre-registered rule, about 7x cheaper).
    """

    def __init__(self, model: str | None = None):
        super().__init__()
        env = os.environ.get("NEUROPT_MODEL", "")
        self._model = model or (env if env.startswith("claude") else DEFAULT_MODEL)
        self._client = None
        self._structured = True  # switched off if the model rejects output_config
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0

    @property
    def name(self) -> str:
        return f"ClaudeBackend({self._model})"

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.generate_structured(prompt, max_tokens=max_tokens)

    def generate_structured(self, prompt, system=None, schema=None, max_tokens=1024):
        kwargs = dict(model=self._model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        if schema is not None and self._structured:
            # Passed through extra_body so older SDK versions can send it too
            kwargs["extra_body"] = {"output_config": {
                "format": {"type": "json_schema", "schema": schema}}}
        try:
            message = self._get_client().messages.create(**kwargs)
        except Exception as e:
            if "extra_body" not in kwargs or not _rejects_structured(e):
                raise
            print(f"  [{self._model} does not accept structured outputs; "
                  "continuing with plain JSON instructions]")
            self._structured = False
            del kwargs["extra_body"]
            message = self._get_client().messages.create(**kwargs)
        usage = message.usage
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
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
        # input_tokens excludes cached tokens: writes cost 1.25x, reads 0.1x (5-minute cache)
        return (self.total_input_tokens * input_price
                + getattr(self, "cache_write_tokens", 0) * input_price * 1.25
                + getattr(self, "cache_read_tokens", 0) * input_price * 0.1
                + self.total_output_tokens * output_price) / 1_000_000


def _rejects_structured(error):
    """True if an API error says the request's output format isn't supported."""
    status = getattr(error, "status_code", None)
    text = str(error).lower()
    return status == 400 and ("output_config" in text or "output format" in text
                              or "structured" in text or "json_schema" in text)
