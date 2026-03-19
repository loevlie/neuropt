"""LLM backend auto-detection and loading."""

from neuropt.backends.base import BaseLLMBackend


def get_default_backend():
    """Auto-detect the best available LLM backend.

    Priority: Claude API -> OpenAI API -> local Qwen -> None (random fallback).
    """
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from neuropt.backends.claude_backend import ClaudeBackend
            backend = ClaudeBackend()
            if backend.is_available():
                return backend
        except Exception:
            pass

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from neuropt.backends.openai_backend import OpenAIBackend
            backend = OpenAIBackend()
            if backend.is_available():
                return backend
        except Exception:
            pass

    try:
        from neuropt.backends.local_qwen import QwenBackend
        backend = QwenBackend()
        if backend.is_available():
            return backend
    except Exception:
        pass

    return None


def get_backend_by_name(name):
    """Get a specific backend by name."""
    if name == "none":
        return None
    elif name == "claude":
        from neuropt.backends.claude_backend import ClaudeBackend
        return ClaudeBackend()
    elif name == "openai":
        from neuropt.backends.openai_backend import OpenAIBackend
        return OpenAIBackend()
    elif name == "qwen":
        from neuropt.backends.local_qwen import QwenBackend
        return QwenBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}. Use 'claude', 'openai', 'qwen', or 'none'.")


__all__ = ["BaseLLMBackend", "get_default_backend", "get_backend_by_name"]
