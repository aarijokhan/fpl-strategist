"""Tests for the LLM factory and Langfuse wiring."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from fpl_strategist.llm import get_chat_model, _build_callbacks


class TestBuildCallbacks:
    def test_returns_none_when_no_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _build_callbacks() is None

    def test_returns_none_when_partial_env_vars(self):
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-test"}, clear=True):
            assert _build_callbacks() is None

    def test_returns_handler_when_both_keys_set(self):
        env = {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
        with patch.dict(os.environ, env, clear=True):
            callbacks = _build_callbacks()
            assert callbacks is not None
            assert len(callbacks) == 1
            assert "CallbackHandler" in type(callbacks[0]).__name__


class TestGetChatModel:
    def test_openai_provider_returns_chatopenai(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            model = get_chat_model(model="gpt-4o", provider="openai")
            assert type(model).__name__ == "ChatOpenAI"

    def test_anthropic_provider_returns_chatanthropic(self):
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            model = get_chat_model(model="claude-sonnet-4-20250514", provider="anthropic")
            assert type(model).__name__ == "ChatAnthropic"

    def test_anthropic_provider_overrides_default_model(self):
        """When provider is anthropic but model is still the default gpt-4o, swap it."""
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            model = get_chat_model(model="gpt-4o", provider="anthropic")
            assert type(model).__name__ == "ChatAnthropic"
            assert model.model == "claude-sonnet-4-20250514"

    def test_langfuse_attached_when_env_vars_present(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test-langfuse",
        }
        with patch.dict(os.environ, env, clear=True):
            model = get_chat_model(provider="openai")
            assert model.callbacks is not None
            assert len(model.callbacks) == 1
            assert "CallbackHandler" in type(model.callbacks[0]).__name__

    def test_langfuse_not_attached_when_env_vars_absent(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            model = get_chat_model(provider="openai")
            assert model.callbacks is None
