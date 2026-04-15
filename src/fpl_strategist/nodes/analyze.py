"""analyze_and_propose node — LLM call.

Evaluates squad weaknesses and proposes one transfer (or hold).
Does NOT enforce budget/squad rules — that's the validator's job.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from fpl_strategist.llm import get_chat_model
from fpl_strategist.nodes.schemas import TransferProposal
from fpl_strategist.state import FPLState

MODEL = "gpt-4o-mini"

PROMPT_TEMPLATE = """\
# Role
You are an expert Fantasy Premier League analyst. You evaluate squads and \
propose one transfer (or hold) based purely on football intelligence: form, \
fixtures, expected points, and tactical fit.

# Context
Target gameweek: GW {target_gw}

## Current Squad
{squad_table}

Budget in bank: £{bank}m | Free transfers: {free_transfers}

## Transfer Candidates (pre-filtered, grouped by position)
{candidates_summary}

# Task
Analyse the squad above given current form and upcoming fixtures. \
Propose ONE transfer (player out, player in) that most improves the squad \
for GW {target_gw}, or recommend "hold" if no transfer is clearly beneficial.

Focus on:
- Players in the squad with poor form or tough upcoming fixtures — transfer out. \
Form is average points per game over the last 30 days: 2.0 = poor, 4.0 = average, 7.0+ = elite
- Candidates with strong form and favourable fixtures — transfer in
- Replace like-for-like (same position)

# Rules
Do NOT check or enforce any of the following — they are validated \
deterministically downstream and are NOT your concern:
- Budget constraints (whether the user can afford the transfer)
- Team limits (max 3 players per Premier League team)
- Squad size or formation rules
- Player availability, injury status, or suspensions

Your ONLY job is football intelligence. Propose the best transfer on \
footballing merit alone. The constraint engine will reject invalid proposals \
and ask you to try again if needed.

# Output format
Return a JSON object with these exact fields:
- action: "transfer" or "hold"
- player_out_id: the squad player's ID to transfer out (null if hold)
- player_in_id: the candidate's ID to bring in (null if hold)
- reasoning: 2-3 sentences explaining your footballing logic
"""


def _format_squad_table(squad: list[dict]) -> str:
    """Format the squad as a readable table for the prompt."""
    lines = ["Name (id) | Pos | Team | Form | Price | EP Next | Fixture"]
    lines.append("--- | --- | --- | --- | --- | --- | ---")
    for p in squad:
        fix = p.get("fixture", {})
        opp = fix.get("opponent", "?")
        loc = "H" if fix.get("is_home") else "A"
        diff = fix.get("difficulty", "?")
        fix_str = f"{opp}({loc}) [{diff}]"
        lines.append(
            f"{p['web_name']} (id={p['id']}) | {p.get('position_name', '?')} | "
            f"{p.get('team_name', '?')} | {p['form']} | "
            f"£{p['now_cost'] / 10:.1f}m | {p.get('ep_next', '—')} | {fix_str}"
        )
    return "\n".join(lines)


def _format_candidates(candidates: list[dict]) -> str:
    """Group candidates by position and format for the prompt."""
    by_pos: dict[str, list[dict]] = {}
    for c in candidates:
        pos = c.get("position_name", "UNK")
        by_pos.setdefault(pos, []).append(c)

    sections = []
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        group = by_pos.get(pos, [])
        if not group:
            continue
        lines = [f"### {pos}"]
        for c in group[:10]:  # Top 10 per position to keep prompt concise
            fix = c.get("fixture", {})
            opp = fix.get("opponent", "?")
            loc = "H" if fix.get("is_home") else "A"
            diff = fix.get("difficulty", "?")
            lines.append(
                f"  {c['web_name']} (id={c['id']}) | "
                f"Form: {c['form']} | EP: {c.get('ep_next', '—')} | "
                f"£{c['now_cost'] / 10:.1f}m | {opp}({loc}) [{diff}]"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _force_invalid_proposal(state: FPLState) -> dict:
    """Demo mode: return an unaffordable transfer to exercise the replan loop."""
    candidates = state.get("candidates", [])
    squad = state["current_squad"]

    # Pick the most expensive candidate
    most_expensive = max(candidates, key=lambda c: c["now_cost"])
    target_pos = most_expensive["element_type"]

    # Pair with the cheapest squad player of the same position
    same_pos = [p for p in squad if p["element_type"] == target_pos]
    if not same_pos:
        same_pos = squad
    cheapest = min(same_pos, key=lambda p: p.get("selling_price", p["now_cost"]))

    return {
        "proposed_transfer": {"out": cheapest, "in": most_expensive},
        "transfer_reasoning": (
            "[FORCE-REPLAN] Deliberately proposing an unaffordable transfer "
            "to exercise the replan loop."
        ),
    }


async def analyze_and_propose(state: FPLState) -> dict:
    """Evaluate the squad and propose one transfer using an LLM."""
    # Demo mode: force an invalid proposal to exercise the replan loop
    if state.get("force_replan"):
        return _force_invalid_proposal(state)

    provider = state.get("provider", "openai")
    llm = get_chat_model(model=MODEL, provider=provider)
    structured_llm = llm.with_structured_output(TransferProposal)

    prompt = PROMPT_TEMPLATE.format(
        target_gw=state["target_gw"],
        squad_table=_format_squad_table(state["current_squad"]),
        bank=f"{state['bank'] / 10:.1f}",
        free_transfers=state.get("free_transfers", 1),
        candidates_summary=_format_candidates(state.get("candidates", [])),
    )

    result: TransferProposal = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    if result.action == "hold" or result.player_out_id is None or result.player_in_id is None:
        return {
            "proposed_transfer": None,
            "transfer_reasoning": result.reasoning,
        }

    # Look up full player dicts for the proposal
    squad_by_id = {p["id"]: p for p in state["current_squad"]}
    candidates_by_id = {c["id"]: c for c in state.get("candidates", [])}

    out_player = squad_by_id.get(result.player_out_id)
    in_player = candidates_by_id.get(result.player_in_id)

    # If the LLM picked an ID we can't resolve, treat as hold
    if out_player is None or in_player is None:
        return {
            "proposed_transfer": None,
            "transfer_reasoning": (
                f"LLM proposed out_id={result.player_out_id}, in_id={result.player_in_id} "
                f"but one could not be resolved. Original reasoning: {result.reasoning}"
            ),
        }

    return {
        "proposed_transfer": {"out": out_player, "in": in_player},
        "transfer_reasoning": result.reasoning,
    }
