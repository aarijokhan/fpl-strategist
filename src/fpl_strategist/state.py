"""LangGraph shared state schema for the FPL Transfer Strategist."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class FPLState(TypedDict, total=False):
    """Shared state passed through all graph nodes.

    Fields marked as required (total=True equivalent) are set at graph invocation.
    All other fields are populated by nodes during execution.
    """

    # --- Inputs (set at invocation) ---
    team_id: int
    target_gw: int
    provider: str  # "openai" or "anthropic", default "openai"

    # --- Data layer (set by fetch_context) ---
    current_squad: list[dict]       # 15 players enriched with form, fixtures, price
    bank: int                       # Budget in tenths (15 = £1.5m)
    free_transfers: int             # 1 or 2
    candidates: list[dict]          # Pre-filtered ~60-80 transfer candidates
    fixtures: list[dict]            # Next GW fixtures with difficulty
    teams: dict[int, str]           # team_id -> short_name
    player_form: dict[int, dict]    # player_id -> recent stats summary

    # --- Proposal (set by analyze_and_propose, overwritten by replan_transfer) ---
    proposed_transfer: dict | None  # {"out": player_dict, "in": player_dict} or None (hold)
    transfer_reasoning: str

    # --- Validation (set by validate_constraints) ---
    is_valid: bool
    violations: list[str]           # Human-readable constraint violation messages
    replan_count: int               # Incremented in replan_transfer node only, capped at 2

    # --- Captain (set by select_captain) ---
    captain_pick: dict | None       # {"player_id": int, "name": str, "reasoning": str}
    vice_captain_pick: dict | None

    # --- Final output (set by explain_recommendation) ---
    recommendation: str

    # --- LLM conversation trace ---
    messages: Annotated[list[BaseMessage], add_messages]
