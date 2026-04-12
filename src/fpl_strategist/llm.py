"""LLM factory with optional Langfuse tracing.

Returns a ChatOpenAI or ChatAnthropic instance. If LANGFUSE_PUBLIC_KEY
and LANGFUSE_SECRET_KEY are both set in the environment, a Langfuse
callback handler is attached automatically. Otherwise, the model is
returned without tracing — tests work without Langfuse keys.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


def get_chat_model(
    model: str = "gpt-4o",
    provider: str = "openai",
) -> BaseChatModel:
    """Create a chat model with optional Langfuse tracing.

    Args:
        model: Model name/ID (e.g. "gpt-4o", "claude-sonnet-4-20250514").
        provider: "openai" or "anthropic".

    Returns:
        A configured BaseChatModel instance.
    """
    callbacks = _build_callbacks()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if model == "gpt-4o":
            model = "claude-sonnet-4-20250514"
        return ChatAnthropic(model=model, callbacks=callbacks)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, callbacks=callbacks)


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
