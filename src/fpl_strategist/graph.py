"""LangGraph graph definition for the FPL Transfer Strategist.

Wires all nodes, edges, and the conditional replan loop.
The replan loop is the architectural centerpiece: validate_constraints
can route back to replan_transfer up to 2 times before falling back
to holding the transfer.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from fpl_strategist.nodes.analyze import analyze_and_propose
from fpl_strategist.nodes.explain import explain_recommendation
from fpl_strategist.nodes.fetch_context import fetch_context
from fpl_strategist.nodes.replan import replan_transfer
from fpl_strategist.nodes.select_captain import select_captain
from fpl_strategist.nodes.validate import validate_constraints
from fpl_strategist.state import FPLState

MAX_REPLANS = 2


def _route_after_validation(state: FPLState) -> str:
    """Conditional router after validate_constraints.

    - Valid proposal → select_captain
    - Invalid, retries < 2 → replan_transfer
    - Invalid, retries >= 2 → select_captain (fallback: hold)
    """
    if state.get("is_valid", False):
        return "select_captain"
    if state.get("replan_count", 0) < MAX_REPLANS:
        return "replan_transfer"
    # Exhausted replans — clear proposal and proceed
    return "select_captain"


def _clear_proposal_if_exhausted(state: FPLState) -> dict:
    """If entering select_captain after exhausted replans, clear the proposal."""
    if not state.get("is_valid", False) and state.get("replan_count", 0) >= MAX_REPLANS:
        return {
            "proposed_transfer": None,
            "transfer_reasoning": (
                "After multiple attempts, no valid transfer could be found "
                "within constraints. Recommendation: hold the free transfer."
            ),
        }
    return {}


async def select_captain_with_fallback(state: FPLState) -> dict:
    """Wrapper that clears proposal if replans exhausted, then selects captain."""
    cleared = _clear_proposal_if_exhausted(state)
    captain_result = await select_captain(state)
    return {**cleared, **captain_result}


def build_graph() -> StateGraph:
    """Construct and return the compiled FPL strategist graph."""
    builder = StateGraph(FPLState)

    # Register nodes
    builder.add_node("fetch_context", fetch_context)
    builder.add_node("analyze_and_propose", analyze_and_propose)
    builder.add_node("validate_constraints", validate_constraints)
    builder.add_node("replan_transfer", replan_transfer)
    builder.add_node("select_captain", select_captain_with_fallback)
    builder.add_node("explain_recommendation", explain_recommendation)

    # Linear edges
    builder.add_edge(START, "fetch_context")
    builder.add_edge("fetch_context", "analyze_and_propose")
    builder.add_edge("analyze_and_propose", "validate_constraints")

    # Conditional: the replan loop
    builder.add_conditional_edges(
        "validate_constraints",
        _route_after_validation,
        {
            "select_captain": "select_captain",
            "replan_transfer": "replan_transfer",
        },
    )

    # Replan loops back to validate
    builder.add_edge("replan_transfer", "validate_constraints")

    # After captain, explain, then end
    builder.add_edge("select_captain", "explain_recommendation")
    builder.add_edge("explain_recommendation", END)

    return builder.compile()


# Pre-built graph instance for import
graph = build_graph()
