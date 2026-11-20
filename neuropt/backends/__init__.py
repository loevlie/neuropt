"""LLM backend auto-detection and loading."""

import os

from neuropt.backends.base import BaseDecisionBackend, BaseLLMBackend
from neuropt.backends.decision_backend import DecisionBackend


def _make_claude(model=None):
    from neuropt.backends.claude_backend import ClaudeBackend
    return ClaudeBackend(model=model)


def _make_openai(model=None):
    from neuropt.backends.openai_backend import OpenAIBackend
    # Plain OpenAI: ignore a base_url meant for the openai-compatible backend
    return OpenAIBackend(model=model, base_url="")


def _make_compatible(model=None):
    from neuropt.backends.openai_backend import OpenAIBackend
    if not os.environ.get("NEUROPT_LLM_BASE_URL"):
        raise ValueError("the openai-compatible backend needs $NEUROPT_LLM_BASE_URL "
                         "(e.g. http://localhost:11434/v1 for Ollama)")
    return OpenAIBackend(model=model)


def _make_qwen(model=None):
    from neuropt.backends.local_qwen import QwenBackend
    return QwenBackend()


# Auto-detection order: first available wins. The local Qwen backend is not
# auto-detected (it downloads 3 GB and runs on CPU); ask for it by name.
_AUTO = ("openai-compatible", "claude", "openai")
_REGISTRY = {
    "claude": _make_claude,
    "openai": _make_openai,
    "openai-compatible": _make_compatible,
    "qwen": _make_qwen,
}


def get_default_backend(model=None):
    """Auto-detect an LLM backend, or None (random search) if there is none.

    Order: an OpenAI-compatible server if ``$NEUROPT_LLM_BASE_URL`` is set,
    then Claude (``$ANTHROPIC_API_KEY``), then OpenAI (``$OPENAI_API_KEY``).
    ``model`` (or ``$NEUROPT_MODEL``) picks the model; a ``claude-*`` model
    selects the Claude backend.
    """
    model = model or os.environ.get("NEUROPT_MODEL") or None
    order = _AUTO
    if model and model.startswith("claude"):
        order = ("claude",)
    elif model:
        order = ("openai-compatible", "openai")
    for name in order:
        try:
            backend = _REGISTRY[name](model)
            if backend.is_available():
                return backend
        except Exception:
            pass
    return None


def get_backend_by_name(name, model=None):
    """Get a specific backend by name, optionally with a model id."""
    if name == "none":
        return None
    if name == "decision":
        return DecisionBackend()
    factory = _REGISTRY.get(name)
    if factory is None:
        valid = ", ".join(f"'{k}'" for k in _REGISTRY) + ", 'decision', or 'none'"
        raise ValueError(f"Unknown backend: {name!r}. Use {valid}.")
    return factory(model)


__all__ = [
    "BaseLLMBackend", "BaseDecisionBackend", "DecisionBackend",
    "get_default_backend", "get_backend_by_name",
]
