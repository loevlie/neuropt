"""LLM backend auto-detection and loading."""

from neuropt.backends.base import BaseLLMBackend


def _make_claude():
    from neuropt.backends.claude_backend import ClaudeBackend
    return ClaudeBackend()


def _make_openai():
    from neuropt.backends.openai_backend import OpenAIBackend
    return OpenAIBackend()


def _make_qwen():
    from neuropt.backends.local_qwen import QwenBackend
    return QwenBackend()


# Auto-detection priority order: first available wins.
_REGISTRY = {
    "claude": _make_claude,
    "openai": _make_openai,
    "qwen": _make_qwen,
}


def get_default_backend():
    """Auto-detect the best available LLM backend.

    Priority: Claude API -> OpenAI API -> local Qwen -> None (random fallback).
    """
    for factory in _REGISTRY.values():
        try:
            backend = factory()
            if backend.is_available():
                return backend
        except Exception:
            pass
    return None


def get_backend_by_name(name):
    """Get a specific backend by name."""
    if name == "none":
        return None
    factory = _REGISTRY.get(name)
    if factory is None:
        valid = ", ".join(f"'{k}'" for k in _REGISTRY) + ", or 'none'"
        raise ValueError(f"Unknown backend: {name!r}. Use {valid}.")
    return factory()


__all__ = ["BaseLLMBackend", "get_default_backend", "get_backend_by_name"]
