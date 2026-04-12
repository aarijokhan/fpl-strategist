"""analyze_and_propose node — LLM call.

Evaluates squad weaknesses and proposes one transfer (or hold).
Does NOT enforce budget/squad rules — that's the validator's job.

This is a placeholder for Phase 3; real LLM integration in Phase 4.
"""

from __future__ import annotations

from fpl_strategist.state import FPLState


async def analyze_and_propose(state: FPLState) -> dict:
    """Propose a transfer based on squad analysis. Placeholder for Phase 4."""
    # Phase 4 will replace this with a real LLM call using with_structured_output()
    return {
        "proposed_transfer": None,
        "transfer_reasoning": "Placeholder: no LLM configured yet.",
    }
