"""replan_transfer node — LLM call.

Increments replan_count, injects violation list into prompt,
and proposes a different transfer avoiding the failed constraints.

This is a placeholder for Phase 3; real LLM integration in Phase 4.
"""

from __future__ import annotations

from fpl_strategist.state import FPLState


async def replan_transfer(state: FPLState) -> dict:
    """Replan after constraint violation. Placeholder for Phase 4.

    IMPORTANT: This node increments replan_count. Do not increment it
    in the validator or the router.
    """
    new_count = state.get("replan_count", 0) + 1

    # Phase 4 will replace this with a real LLM call that receives
    # state["violations"] and proposes a different transfer.
    return {
        "replan_count": new_count,
        "proposed_transfer": None,
        "transfer_reasoning": f"Placeholder replan (attempt {new_count}): no LLM configured yet.",
    }
