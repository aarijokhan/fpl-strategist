"""validate_constraints node — deterministic, no LLM.

Runs the constraint engine against the proposed transfer and
sets is_valid / violations in state.
"""

from __future__ import annotations

from fpl_strategist.constraints.engine import validate_transfer
from fpl_strategist.state import FPLState


async def validate_constraints(state: FPLState) -> dict:
    """Validate the proposed transfer against FPL rules."""
    proposed = state.get("proposed_transfer")

    # Hold transfer — always valid
    if proposed is None:
        return {
            "is_valid": True,
            "violations": [],
        }

    result = validate_transfer(
        current_squad=state["current_squad"],
        bank=state["bank"],
        proposed_out=proposed.get("out"),
        proposed_in=proposed.get("in"),
        all_players=None,  # Skip pool existence check — candidates already filtered
        teams=state.get("teams", {}),
    )

    return {
        "is_valid": result.is_valid,
        "violations": result.violations,
    }
