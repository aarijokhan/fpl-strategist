"""Tests for eval/judge.py — LLM-judge coherence scoring."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from fpl_strategist.eval.judge import (
    JudgeScore,
    _format_context,
    judge_coherence,
)


# ---------------------------------------------------------------------------
# Minimal state fixture
# ---------------------------------------------------------------------------

def _make_state(
    transfer_reasoning: str = "Salah (form 8.5) has a home fixture vs Southampton (difficulty 2).",
    captain_reasoning: str = "Haaland is the safest captain with form 7.0 against a weak defence.",
    recommendation: str = "Transfer in Salah for Elanga. Captain Haaland.",
) -> dict:
    return {
        "team_id": 44,
        "target_gw": 15,
        "current_squad": [
            {
                "web_name": "Salah", "position_name": "MID",
                "form_float": 8.5, "fixture": {"difficulty": 2}, "ep_next_float": 7.0,
            },
            {
                "web_name": "Haaland", "position_name": "FWD",
                "form_float": 7.0, "fixture": {"difficulty": 3}, "ep_next_float": 6.5,
            },
        ],
        "proposed_transfer": {
            "out": {"web_name": "Elanga", "form_float": 1.5, "now_cost": 55},
            "in":  {"web_name": "Diaz",   "form_float": 6.5, "now_cost": 80},
        },
        "captain_pick": {"id": 2, "name": "Haaland"},
        "vice_captain_pick": {"id": 1, "name": "Salah"},
        "transfer_reasoning": transfer_reasoning,
        "captain_reasoning": captain_reasoning,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# JudgeScore schema
# ---------------------------------------------------------------------------

class TestJudgeScoreSchema:
    def test_valid_scores_accepted(self):
        score = JudgeScore(
            factual_grounding=4,
            logical_coherence=3,
            actionability=5,
            explanation="Reasoning was well-grounded.",
        )
        assert score.average == pytest.approx(4.0)

    def test_schema_rejects_score_zero(self):
        with pytest.raises(ValidationError):
            JudgeScore(factual_grounding=0, logical_coherence=3, actionability=3, explanation="x")

    def test_schema_rejects_score_six(self):
        with pytest.raises(ValidationError):
            JudgeScore(factual_grounding=6, logical_coherence=3, actionability=3, explanation="x")

    def test_average_property(self):
        score = JudgeScore(factual_grounding=2, logical_coherence=4, actionability=3, explanation="x")
        assert score.average == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_prompt_contains_squad_context(self):
        state = _make_state()
        ctx = _format_context(state)
        assert "Salah" in ctx["squad_summary"]
        assert "Haaland" in ctx["squad_summary"]

    def test_prompt_contains_reasoning(self):
        state = _make_state(transfer_reasoning="Sell Elanga for Diaz due to form.")
        ctx = _format_context(state)
        assert "Sell Elanga for Diaz" in ctx["transfer_reasoning"]

    def test_prompt_contains_fixture_form_context(self):
        state = _make_state()
        ctx = _format_context(state)
        # Squad summary should include form and difficulty values
        assert "8.5" in ctx["squad_summary"]  # Salah's form
        assert "2" in ctx["squad_summary"]    # fixture difficulty


# ---------------------------------------------------------------------------
# judge_coherence (mocked LLM)
# ---------------------------------------------------------------------------

class TestJudgeCoherence:
    async def test_returns_valid_scores(self):
        mock_score = JudgeScore(
            factual_grounding=4, logical_coherence=4, actionability=5,
            explanation="Well-grounded reasoning with specific data."
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = AsyncMock(
            ainvoke=AsyncMock(return_value=mock_score)
        )
        with patch("fpl_strategist.eval.judge.get_chat_model", return_value=mock_llm):
            result = await judge_coherence(_make_state(), provider="openai")

        assert isinstance(result, JudgeScore)
        assert 1 <= result.factual_grounding <= 5
        assert 1 <= result.logical_coherence <= 5
        assert 1 <= result.actionability <= 5

    async def test_uses_anthropic_by_default(self):
        mock_score = JudgeScore(
            factual_grounding=4, logical_coherence=4, actionability=4, explanation="x"
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = AsyncMock(
            ainvoke=AsyncMock(return_value=mock_score)
        )
        with patch("fpl_strategist.eval.judge.get_chat_model", return_value=mock_llm) as mock_factory, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            await judge_coherence(_make_state())
            call_kwargs = mock_factory.call_args
            assert call_kwargs.kwargs.get("provider") == "anthropic" or \
                   call_kwargs.args[1] == "anthropic" if call_kwargs.args else \
                   "anthropic" in str(call_kwargs)

    async def test_falls_back_to_openai_when_no_anthropic_key(self):
        mock_score = JudgeScore(
            factual_grounding=3, logical_coherence=3, actionability=3, explanation="x"
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = AsyncMock(
            ainvoke=AsyncMock(return_value=mock_score)
        )
        env_without_anthropic = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch("fpl_strategist.eval.judge.get_chat_model", return_value=mock_llm) as mock_factory, \
             patch.dict(os.environ, env_without_anthropic, clear=True):
            # Default provider=anthropic but no key → should fall back to openai
            await judge_coherence(_make_state())
            call_kwargs = mock_factory.call_args
            assert "openai" in str(call_kwargs)
