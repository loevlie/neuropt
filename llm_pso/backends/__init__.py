"""LLM backend auto-detection and loading."""

from llm_pso.backends.base import BaseLLMBackend


def get_default_backend():
    """Auto-detect the best available LLM backend.

    Priority: Claude API -> OpenAI API -> local Qwen -> None (pure PSO).
    """
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from llm_pso.backends.claude_backend import ClaudeBackend
            backend = ClaudeBackend()
            if backend.is_available():
                return backend
        except Exception:
            pass

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from llm_pso.backends.openai_backend import OpenAIBackend
            backend = OpenAIBackend()
            if backend.is_available():
                return backend
        except Exception:
            pass

    try:
        from llm_pso.backends.local_qwen import QwenBackend
        backend = QwenBackend()
        if backend.is_available():
            return backend
    except Exception:
        pass

    return None


def get_backend_by_name(name):
    """Get a specific backend by name.

    Args:
        name: One of 'claude', 'openai', 'qwen', 'none'.

    Returns:
        Backend instance or None if name is 'none'.
    """
    if name == "none":
        return None
    elif name == "claude":
        from llm_pso.backends.claude_backend import ClaudeBackend
        return ClaudeBackend()
    elif name == "openai":
        from llm_pso.backends.openai_backend import OpenAIBackend
        return OpenAIBackend()
    elif name == "qwen":
        from llm_pso.backends.local_qwen import QwenBackend
        return QwenBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}. Use 'claude', 'openai', 'qwen', or 'none'.")


__all__ = ["BaseLLMBackend", "get_default_backend", "get_backend_by_name"]
