"""LLM factory with optional Langfuse tracing.

Returns a ChatOpenAI or ChatAnthropic instance. If LANGFUSE_PUBLIC_KEY
and LANGFUSE_SECRET_KEY are both set in the environment, a Langfuse
callback handler is attached automatically. Otherwise, the model is
returned without tracing — tests work without Langfuse keys.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


_api_key_override: str | None = None


def set_api_key_override(key: str | None) -> None:
    """Set a per-run API key override (used by BYOK in the web UI).

    With concurrency_limit=1 on the Gradio handler, only one graph
    executes at a time, so a module-level variable is safe.
    """
    global _api_key_override
    _api_key_override = key


def get_chat_model(
    model: str = "gpt-4o",
    provider: str = "openai",
    api_key: str | None = None,
) -> BaseChatModel:
    """Create a chat model with optional Langfuse tracing.

    Args:
        model: Model name/ID (e.g. "gpt-4o", "claude-sonnet-4-20250514").
        provider: "openai" or "anthropic".
        api_key: Explicit API key. Falls back to _api_key_override, then env.

    Returns:
        A configured BaseChatModel instance.
    """
    key = api_key or _api_key_override
    callbacks = _build_callbacks()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if model == "gpt-4o":
            model = "claude-sonnet-4-20250514"
        kwargs: dict = {"model": model, "callbacks": callbacks}
        if key:
            kwargs["api_key"] = key
        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    kwargs = {"model": model, "callbacks": callbacks}
    if key:
        kwargs["api_key"] = key
    return ChatOpenAI(**kwargs)


def _build_callbacks() -> list | None:
    """Attach Langfuse callback handler if env vars are present."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return None

    from langfuse.langchain import CallbackHandler as LangfuseHandler

    # Langfuse v4 reads LANGFUSE_SECRET_KEY and LANGFUSE_HOST from env vars directly.
    # Only public_key is passed as a constructor arg.
    handler = LangfuseHandler(public_key=public_key)
    return [handler]
