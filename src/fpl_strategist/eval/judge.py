"""LLM-judge coherence scoring for Phase 6 evaluation.

Uses Claude Sonnet (cross-family from the GPT-based agent) to evaluate
whether the agent's reasoning is internally consistent with the data it cites.

Three dimensions, each scored 1-5:
  - Factual Grounding: does the reasoning cite specific, verifiable data points?
  - Logical Coherence: does the conclusion follow from the cited evidence?
  - Actionability: would an FPL manager know exactly what to do?

Cross-family note: the agent uses GPT-4o-mini/GPT-4o; the judge uses Claude
Sonnet to avoid same-model self-evaluation bias.
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

from fpl_strategist.llm import get_chat_model

logger = logging.getLogger(__name__)

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
_FALLBACK_OPENAI_MODEL = "gpt-4o"

JUDGE_SYSTEM_PROMPT = """\
You are evaluating the reasoning quality of an FPL (Fantasy Premier League) \
transfer recommendation system. You are NOT judging whether the transfer was \
a good footballing decision — only whether the REASONING is well-structured, \
factually grounded, and actionable.

## Context provided to the system

Target gameweek: GW {target_gw}

Squad summary (abbreviated):
{squad_summary}

Transfer decision: {transfer_summary}
Captain decision: {captain_summary}

## System's reasoning

Transfer reasoning: {transfer_reasoning}
Captain reasoning: {captain_reasoning}
Final recommendation: {recommendation}

## Scoring rubric

Score each dimension from 1 to 5:

### Factual Grounding (1-5)
5: Cites specific form values, fixture difficulties, opponent names, prices, \
and expected points from the context
4: Cites most relevant data points; minor omissions
3: Cites some data but misses key factors or uses vague language
2: Mostly generic statements with rare specific citations
1: No specific data cited, or fabricates statistics not in the context

### Logical Coherence (1-5)
5: Conclusion follows directly from cited evidence; no contradictions
4: Mostly coherent with minor logical gaps
3: Some reasoning connects but key logical steps are missing
2: Conclusion weakly supported by evidence; contradictions present
1: Conclusion contradicts cited evidence, or reasoning is circular

### Actionability (1-5)
5: Reader knows exactly which player to transfer, who to captain, and why
4: Clear recommendation with minor ambiguity
3: Recommendation is present but lacks specificity in one area
2: Vague recommendation that requires interpretation
1: No clear action, or contradictory instructions

## Output

Return a JSON object with exactly these fields:
- factual_grounding: integer 1-5
- logical_coherence: integer 1-5
- actionability: integer 1-5
- explanation: 2-3 sentences justifying your scores, citing specific examples \
from the reasoning that informed each score
"""


class JudgeScore(BaseModel):
    """Structured output from the LLM judge."""

    factual_grounding: int = Field(ge=1, le=5)
    logical_coherence: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    explanation: str

    @property
    def average(self) -> float:
        """Mean of the three dimension scores."""
        return (self.factual_grounding + self.logical_coherence + self.actionability) / 3


def _format_squad_summary(squad: list[dict]) -> str:
    """Abbreviate squad to a compact table for the judge prompt."""
    lines = ["Name | Pos | Form | Fixture Diff | EP Next"]
    for p in squad[:11]:  # starting XI only to keep prompt short
        pos = p.get("position_name", "???")
        form = p.get("form_float", 0.0)
        diff = p.get("fixture", {}).get("difficulty", "?")
        ep = p.get("ep_next_float", 0.0)
        lines.append(f"{p['web_name']} | {pos} | {form} | {diff} | {ep}")
    return "\n".join(lines)


def _format_context(state: dict) -> dict[str, str]:
    """Extract and format template variables from an FPLState dict."""
    target_gw = state.get("target_gw", "?")
    squad = state.get("current_squad", [])
    transfer = state.get("proposed_transfer")
    captain = state.get("captain_pick", {}) or {}
    vice = state.get("vice_captain_pick", {}) or {}

    if transfer:
        out_p = transfer.get("out", {})
        in_p = transfer.get("in", {})
        transfer_summary = (
            f"Sell {out_p.get('web_name', '?')} "
            f"(form {out_p.get('form_float', '?')}, £{out_p.get('now_cost', 0)/10:.1f}m) → "
            f"Buy {in_p.get('web_name', '?')} "
            f"(form {in_p.get('form_float', '?')}, £{in_p.get('now_cost', 0)/10:.1f}m)"
        )
    else:
        transfer_summary = "Hold — no transfer this gameweek."

    captain_summary = (
        f"Captain: {captain.get('name', '?')}, "
        f"Vice-captain: {vice.get('name', '?')}"
    )

    return {
        "target_gw": str(target_gw),
        "squad_summary": _format_squad_summary(squad),
        "transfer_summary": transfer_summary,
        "captain_summary": captain_summary,
        "transfer_reasoning": state.get("transfer_reasoning", "(not available)"),
        "captain_reasoning": state.get("captain_reasoning", "(not available)"),
        "recommendation": state.get("recommendation", "(not available)"),
    }


async def judge_coherence(
    state: dict,
    *,
    provider: str = "anthropic",
    model: str | None = None,
) -> JudgeScore:
    """Score the coherence of the agent's reasoning using an LLM judge.

    Uses Claude Sonnet by default (cross-family from the GPT-based agent).
    Falls back to gpt-4o if ANTHROPIC_API_KEY is not set.

    Args:
        state: The full FPLState dict after graph execution.
        provider: "anthropic" (default) or "openai".
        model: Override the model name. Uses sensible defaults if None.

    Returns:
        JudgeScore with per-dimension scores and explanation.
    """
    # Fall back to openai if anthropic key is absent
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY not set — falling back to openai for judge coherence scoring."
        )
        provider = "openai"

    if model is None:
        model = _DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else _FALLBACK_OPENAI_MODEL

    llm = get_chat_model(model=model, provider=provider)
    structured_llm = llm.with_structured_output(JudgeScore)

    context = _format_context(state)
    prompt = JUDGE_SYSTEM_PROMPT.format(**context)

    return await structured_llm.ainvoke(prompt)
