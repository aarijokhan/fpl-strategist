"""explain_recommendation node — LLM call.

Synthesizes the transfer decision, captain pick, and reasoning
into a final natural-language recommendation.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from fpl_strategist.llm import get_chat_model
from fpl_strategist.state import FPLState

MODEL = "gpt-4o"

PROMPT_TEMPLATE = """\
# Role
You are an FPL content creator writing a weekly transfer and captaincy \
recommendation. Your voice is like FPL Andy from Let's Talk FPL: direct, \
specific, decisive, assumes the reader knows the basics of FPL. No hedging, \
no filler, no meta-commentary about being an AI or an agent.

# Context
Target gameweek: GW {target_gw}

## Transfer Decision
{transfer_section}

## Transfer Reasoning
{transfer_reasoning}

## Replan History
{replan_section}

## Captaincy
Captain: {captain_name}
Vice-captain: {vice_captain_name}

Captain reasoning: {captain_reasoning}

# Task
Write a 3-5 paragraph recommendation that an FPL manager would actually read \
and act on.

Structure:
1. Lead with the decision — the transfer and the captain pick, stated \
confidently and clearly.
2. Explain why — the footballing logic behind the transfer. Cite specific \
form values, expected points, fixture difficulties, and opponent names. \
Do not summarise away the specificity that the data provides.
3. Explain the captaincy choice with the same specificity — form, fixture, \
track record of returns.
4. If a replan occurred, briefly acknowledge it: what was initially proposed, \
why it was rejected (cite the specific constraint violation), and why the \
revised pick is sound. Frame this as honesty about the process, not as a \
weakness.
5. Close with any caveats or what to watch for — e.g. a press conference that \
could change the calculus, or a player worth monitoring for next week.

# Style
- Authoritative and conversational. Confident, not hedging.
- No generic filler like "based on the analysis above" or "after careful \
consideration."
- Cite specific numbers: form values, EP next, fixture difficulties, prices.
- If the decision is to hold, explain what was considered, why no transfer \
was clearly better, and what to watch for next gameweek.
- Do not refer to yourself as an AI, agent, model, or system.
- Do not use bullet points or headers — write flowing prose paragraphs.

# Output format
Return ONLY the recommendation text — no JSON wrapper, no preamble, no \
"Here is my recommendation:" lead-in. Just the paragraphs.
"""


def _build_transfer_section(state: FPLState) -> str:
    """Format the transfer decision for the prompt."""
    transfer = state.get("proposed_transfer")
    if transfer is None:
        return "Decision: HOLD — no transfer this gameweek."

    out_p = transfer["out"]
    in_p = transfer["in"]
    return (
        f"Transfer OUT: {out_p['web_name']} "
        f"(£{out_p['now_cost'] / 10:.1f}m, form {out_p['form']})\n"
        f"Transfer IN: {in_p['web_name']} "
        f"(£{in_p['now_cost'] / 10:.1f}m, form {in_p['form']})"
    )


def _build_replan_section(state: FPLState) -> str:
    """Format replan history for the prompt."""
    replan_count = state.get("replan_count", 0)
    if replan_count == 0:
        return "No replan occurred — the initial proposal passed all constraints."

    rejected = state.get("rejected_proposals", [])
    violations_history = state.get("replan_violations_history", [])

    lines = [f"The constraint validator rejected {replan_count} proposal(s) before the final pick."]
    for i, (proposal, violations) in enumerate(
        zip(rejected, violations_history), start=1,
    ):
        out_name = proposal.get("out", {}).get("web_name", "Unknown")
        in_name = proposal.get("in", {}).get("web_name", "Unknown")
        v_str = "; ".join(violations) if violations else "unknown violation"
        lines.append(
            f"Attempt {i}: {out_name} → {in_name} — rejected because: {v_str}"
        )

    return "\n".join(lines)


async def explain_recommendation(state: FPLState) -> dict:
    """Synthesize all decisions into a natural-language recommendation."""
    provider = state.get("provider", "openai")
    llm = get_chat_model(model=MODEL, provider=provider)

    captain = state.get("captain_pick") or {}
    vice = state.get("vice_captain_pick") or {}

    prompt = PROMPT_TEMPLATE.format(
        target_gw=state["target_gw"],
        transfer_section=_build_transfer_section(state),
        transfer_reasoning=state.get("transfer_reasoning", "No reasoning provided."),
        replan_section=_build_replan_section(state),
        captain_name=captain.get("name", "Unknown"),
        vice_captain_name=vice.get("name", "Unknown"),
        captain_reasoning=state.get("captain_reasoning", "No reasoning provided."),
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])

    return {"recommendation": response.content}
