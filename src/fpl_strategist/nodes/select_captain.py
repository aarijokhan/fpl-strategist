"""select_captain node — LLM call.

Picks captain and vice-captain from the post-transfer squad
based on fixture difficulty and recent form.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from fpl_strategist.llm import get_chat_model
from fpl_strategist.nodes.analyze import _format_squad_table
from fpl_strategist.nodes.schemas import CaptainPick
from fpl_strategist.state import FPLState

MODEL = "gpt-4o-mini"

PROMPT_TEMPLATE = """\
# Role
You are an expert Fantasy Premier League captain selector. You pick a captain \
and vice-captain for the upcoming gameweek based on form, fixture difficulty, \
and expected returns.

# Context
Target gameweek: GW {target_gw}

## Post-Transfer Squad
{squad_table}

# Task
Pick a captain and a vice-captain from the squad above for GW {target_gw}. \
The captain's points are doubled, so choose the player most likely to deliver \
a big haul this gameweek.

Prioritize:
- High recent form (players on a scoring streak)
- Favorable upcoming fixture difficulty (lower number = easier match)
- Track record of heavy returns: goals and assists for attackers and \
midfielders, clean sheets and bonus points for defenders and goalkeepers
- The vice-captain is a backup if the captain doesn't play — ideally pick \
one from a different fixture than the captain

# Rules
- Captain and vice-captain must be different players
- Prefer players expected to start (not on the bench), but you may pick \
from any of the 15 squad players

# Output format
Return a JSON object with these exact fields:
- captain_id: the squad player's ID for captain
- vice_captain_id: a different squad player's ID for vice-captain
- reasoning: 2-3 sentences citing specific form numbers and fixture difficulty
"""


def _build_post_transfer_squad(state: FPLState) -> list[dict]:
    """Build the squad after the proposed transfer (if any)."""
    squad = list(state["current_squad"])
    transfer = state.get("proposed_transfer")
    if transfer is not None:
        out_id = transfer["out"]["id"]
        squad = [p for p in squad if p["id"] != out_id]
        squad.append(transfer["in"])
    return squad


async def select_captain(state: FPLState) -> dict:
    """Select captain and vice-captain from the post-transfer squad."""
    provider = state.get("provider", "openai")
    llm = get_chat_model(model=MODEL, provider=provider)
    structured_llm = llm.with_structured_output(CaptainPick)

    post_squad = _build_post_transfer_squad(state)

    prompt = PROMPT_TEMPLATE.format(
        target_gw=state["target_gw"],
        squad_table=_format_squad_table(post_squad),
    )

    result: CaptainPick = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    # Resolve names from the post-transfer squad
    squad_by_id = {p["id"]: p for p in post_squad}
    cap_player = squad_by_id.get(result.captain_id, {})
    vc_player = squad_by_id.get(result.vice_captain_id, {})

    return {
        "captain_pick": {
            "id": result.captain_id,
            "name": cap_player.get("web_name", "Unknown"),
        },
        "vice_captain_pick": {
            "id": result.vice_captain_id,
            "name": vc_player.get("web_name", "Unknown"),
        },
        "captain_reasoning": result.reasoning,
    }
