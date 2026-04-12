"""select_captain node — LLM call.

Picks captain and vice-captain from the post-transfer squad
based on fixture difficulty and recent form.

This is a placeholder for Phase 3; real LLM integration in Phase 4.
"""

from __future__ import annotations

from fpl_strategist.state import FPLState


async def select_captain(state: FPLState) -> dict:
    """Select captain and vice-captain. Placeholder for Phase 4."""
    # Phase 4 will replace this with a real LLM call using with_structured_output()
    return {
        "captain_pick": None,
        "vice_captain_pick": None,
    }
