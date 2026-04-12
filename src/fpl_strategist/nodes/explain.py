"""explain_recommendation node — LLM call.

Synthesizes the transfer decision, captain pick, and reasoning
into a final natural-language recommendation.

This is a placeholder for Phase 3; real LLM integration in Phase 4.
"""

from __future__ import annotations

from fpl_strategist.state import FPLState


async def explain_recommendation(state: FPLState) -> dict:
    """Generate final recommendation. Placeholder for Phase 4."""
    # Phase 4 will replace this with a real LLM call
    return {
        "recommendation": "Placeholder recommendation: LLM not configured yet.",
    }
