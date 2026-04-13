"""replan_transfer node — LLM call.

Increments replan_count, injects violation list into prompt,
and proposes a different transfer avoiding the failed constraints.

IMPORTANT: replan_count is incremented in this node only.
Do not increment it in the validator or the router.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from fpl_strategist.llm import get_chat_model
from fpl_strategist.nodes.analyze import _format_candidates, _format_squad_table
from fpl_strategist.nodes.schemas import TransferProposal
from fpl_strategist.state import FPLState

MODEL = "gpt-4o-mini"

PROMPT_TEMPLATE = """\
# Role
You are an expert Fantasy Premier League analyst. Your previous transfer \
proposal was rejected by the constraint validator. You must propose a \
DIFFERENT transfer that avoids the specific violations listed below.

# Context
Target gameweek: GW {target_gw}

## Current Squad
{squad_table}

Budget in bank: £{bank}m | Free transfers: {free_transfers}

## Transfer Candidates (pre-filtered, grouped by position)
{candidates_summary}

# Previous attempt
You proposed: {out_name} (id={out_id}) OUT → {in_name} (id={in_id}) IN
Reasoning: {previous_reasoning}

# Violations
The constraint validator rejected your proposal for these reasons:
{numbered_violations}

# Task
Propose a DIFFERENT transfer — at minimum a different player_in_id, and \
optionally a different player_out_id — that avoids the violations listed \
above. If no reasonable alternative transfer exists given the constraints, \
output "hold" (action="hold", both IDs null).

Focus on:
- Avoiding the specific constraints that were violated
- Finding the next-best candidate on footballing merit
- Same position replacement (like-for-like)

# Rules
Do NOT check or enforce any of the following — they are validated \
deterministically downstream and are NOT your concern:
- Budget constraints (whether the user can afford the transfer)
- Team limits (max 3 players per Premier League team)
- Squad size or formation rules
- Player availability, injury status, or suspensions

The deterministic validator runs again after this node. If your revised \
proposal also violates constraints, you will be asked to try yet again \
(up to 2 total replans).

# Output format
Return a JSON object with these exact fields:
- action: "transfer" or "hold"
- player_out_id: the squad player's ID to transfer out (null if hold)
- player_in_id: the candidate's ID to bring in (null if hold)
- reasoning: 2-3 sentences explaining your revised logic and how it avoids the violation
"""


def _format_violations(violations: list[str]) -> str:
    """Format violations as a numbered list."""
    return "\n".join(f"{i + 1}. {v}" for i, v in enumerate(violations))


async def replan_transfer(state: FPLState) -> dict:
    """Replan after constraint violation.

    IMPORTANT: This node increments replan_count. Do not increment it
    in the validator or the router.
    """
    new_count = state.get("replan_count", 0) + 1

    provider = state.get("provider", "openai")
    llm = get_chat_model(model=MODEL, provider=provider)
    structured_llm = llm.with_structured_output(TransferProposal)

    # Extract previous proposal details
    prev = state.get("proposed_transfer") or {}
    out_player = prev.get("out", {})
    in_player = prev.get("in", {})

    prompt = PROMPT_TEMPLATE.format(
        target_gw=state["target_gw"],
        squad_table=_format_squad_table(state["current_squad"]),
        bank=f"{state['bank'] / 10:.1f}",
        free_transfers=state.get("free_transfers", 1),
        candidates_summary=_format_candidates(state.get("candidates", [])),
        out_name=out_player.get("web_name", "Unknown"),
        out_id=out_player.get("id", "?"),
        in_name=in_player.get("web_name", "Unknown"),
        in_id=in_player.get("id", "?"),
        previous_reasoning=state.get("transfer_reasoning", "No reasoning provided."),
        numbered_violations=_format_violations(state.get("violations", [])),
    )

    result: TransferProposal = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    if result.action == "hold" or result.player_out_id is None or result.player_in_id is None:
        return {
            "proposed_transfer": None,
            "transfer_reasoning": result.reasoning,
            "replan_count": new_count,
        }

    # Look up full player dicts
    squad_by_id = {p["id"]: p for p in state["current_squad"]}
    candidates_by_id = {c["id"]: c for c in state.get("candidates", [])}

    resolved_out = squad_by_id.get(result.player_out_id)
    resolved_in = candidates_by_id.get(result.player_in_id)

    if resolved_out is None or resolved_in is None:
        return {
            "proposed_transfer": None,
            "transfer_reasoning": (
                f"LLM proposed out_id={result.player_out_id}, in_id={result.player_in_id} "
                f"but one could not be resolved. Original reasoning: {result.reasoning}"
            ),
            "replan_count": new_count,
        }

    return {
        "proposed_transfer": {"out": resolved_out, "in": resolved_in},
        "transfer_reasoning": result.reasoning,
        "replan_count": new_count,
    }
