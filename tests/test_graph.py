"""Tests for graph routing logic and the replan loop.

These tests verify the graph structure without hitting the FPL API
or making LLM calls — they test routing, replan count behavior,
and the fallback-to-hold path.
"""

from __future__ import annotations

import pytest

from fpl_strategist.graph import (
    MAX_REPLANS,
    _clear_proposal_if_exhausted,
    _route_after_validation,
)


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestRouteAfterValidation:
    def test_valid_routes_to_captain(self):
        state = {"is_valid": True, "replan_count": 0}
        assert _route_after_validation(state) == "select_captain"

    def test_valid_routes_to_captain_even_with_replans(self):
        state = {"is_valid": True, "replan_count": 2}
        assert _route_after_validation(state) == "select_captain"

    def test_invalid_first_attempt_routes_to_replan(self):
        state = {"is_valid": False, "replan_count": 0}
        assert _route_after_validation(state) == "replan_transfer"

    def test_invalid_second_attempt_routes_to_replan(self):
        state = {"is_valid": False, "replan_count": 1}
        assert _route_after_validation(state) == "replan_transfer"

    def test_invalid_exhausted_routes_to_captain(self):
        state = {"is_valid": False, "replan_count": 2}
        assert _route_after_validation(state) == "select_captain"

    def test_invalid_over_max_routes_to_captain(self):
        state = {"is_valid": False, "replan_count": 5}
        assert _route_after_validation(state) == "select_captain"

    def test_defaults_when_keys_missing(self):
        """Missing keys should not crash — defaults to replan path."""
        state = {}
        assert _route_after_validation(state) == "replan_transfer"


# ---------------------------------------------------------------------------
# Fallback proposal clearing
# ---------------------------------------------------------------------------

class TestClearProposalIfExhausted:
    def test_clears_when_invalid_and_exhausted(self):
        state = {"is_valid": False, "replan_count": 2, "proposed_transfer": {"out": {}, "in": {}}}
        result = _clear_proposal_if_exhausted(state)
        assert result["proposed_transfer"] is None
        assert "hold" in result["transfer_reasoning"].lower()

    def test_does_not_clear_when_valid(self):
        state = {"is_valid": True, "replan_count": 2, "proposed_transfer": {"out": {}, "in": {}}}
        result = _clear_proposal_if_exhausted(state)
        assert result == {}

    def test_does_not_clear_when_retries_remaining(self):
        state = {"is_valid": False, "replan_count": 1, "proposed_transfer": {"out": {}, "in": {}}}
        result = _clear_proposal_if_exhausted(state)
        assert result == {}


# ---------------------------------------------------------------------------
# Replan count behavior
# ---------------------------------------------------------------------------

class TestReplanCountInvariant:
    def test_max_replans_is_two(self):
        """PRD specifies replan cap at 2."""
        assert MAX_REPLANS == 2

    @pytest.mark.asyncio
    async def test_replan_node_increments_count(self):
        """replan_transfer must be the only place that increments replan_count."""
        from fpl_strategist.nodes.replan import replan_transfer

        state = {"replan_count": 0, "violations": ["test violation"]}
        result = await replan_transfer(state)
        assert result["replan_count"] == 1

        state2 = {"replan_count": 1, "violations": ["test violation"]}
        result2 = await replan_transfer(state2)
        assert result2["replan_count"] == 2

    @pytest.mark.asyncio
    async def test_validate_node_does_not_increment_count(self):
        """validate_constraints must NOT touch replan_count."""
        from fpl_strategist.nodes.validate import validate_constraints

        state = {
            "current_squad": [
                {"id": 1, "web_name": "P1", "element_type": 3, "team": 1,
                 "now_cost": 80, "selling_price": 80, "status": "a"},
            ],
            "bank": 15,
            "proposed_transfer": None,
            "teams": {},
        }
        result = await validate_constraints(state)
        assert "replan_count" not in result


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

class TestGraphCompilation:
    def test_graph_compiles(self):
        """The graph should compile without errors."""
        from fpl_strategist.graph import graph
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from fpl_strategist.graph import build_graph
        g = build_graph()
        node_names = set(g.get_graph().nodes.keys())
        expected = {
            "__start__", "__end__",
            "fetch_context", "analyze_and_propose",
            "validate_constraints", "replan_transfer",
            "select_captain", "explain_recommendation",
        }
        assert expected == node_names
