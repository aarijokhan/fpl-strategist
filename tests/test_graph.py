"""Tests for graph routing logic, the replan loop, and end-to-end integration.

Unit tests verify the graph structure without hitting the FPL API
or making LLM calls — they test routing, replan count behavior,
and the fallback-to-hold path.

End-to-end tests invoke the compiled graph with deterministic stubs
patched in for all LLM/API nodes, verifying the full happy-path,
single-replan, and exhausted-replan flows.
"""

from __future__ import annotations

from unittest.mock import patch

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
    async def test_replan_node_increments_count(self, sample_populated_state):
        """replan_transfer must be the only place that increments replan_count."""
        from unittest.mock import AsyncMock, patch

        from fpl_strategist.nodes.replan import replan_transfer
        from fpl_strategist.nodes.schemas import TransferProposal

        mock_response = TransferProposal(
            action="hold", player_out_id=None, player_in_id=None,
            reasoning="Mock: no valid alternative.",
        )
        mock_llm = AsyncMock(return_value=mock_response)

        state = {**sample_populated_state, "replan_count": 0, "violations": ["test"]}
        with patch("fpl_strategist.nodes.replan.get_chat_model") as mock_gcm:
            mock_gcm.return_value.with_structured_output.return_value.ainvoke = mock_llm
            result = await replan_transfer(state)
        assert result["replan_count"] == 1

        state2 = {**sample_populated_state, "replan_count": 1, "violations": ["test"]}
        with patch("fpl_strategist.nodes.replan.get_chat_model") as mock_gcm:
            mock_gcm.return_value.with_structured_output.return_value.ainvoke = mock_llm
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


# ---------------------------------------------------------------------------
# End-to-end integration tests (deterministic stubs, no LLM/API)
# ---------------------------------------------------------------------------

class TestGraphEndToEnd:
    """Invoke the compiled graph with patched nodes to verify full flows."""

    @pytest.mark.asyncio
    async def test_happy_path_no_replan(self, sample_populated_state):
        """Valid proposal on first try — no replan loop fires."""
        squad = sample_populated_state["current_squad"]
        candidates = sample_populated_state["candidates"]

        async def stub_fetch(state):
            return {
                "current_squad": squad, "bank": 25, "free_transfers": 1,
                "candidates": candidates, "fixtures": [],
                "teams": sample_populated_state["teams"], "player_form": {},
                "replan_count": 0, "proposed_transfer": None,
                "is_valid": False, "violations": [],
            }

        async def stub_analyze(state):
            return {
                "proposed_transfer": {"out": squad[11], "in": candidates[4]},
                "transfer_reasoning": "Elanga out, Diaz in.",
            }

        async def stub_validate(state):
            return {"is_valid": True, "violations": []}

        async def stub_captain(state):
            return {"captain_pick": {"player_id": 300}, "vice_captain_pick": {"player_id": 400}}

        async def stub_explain(state):
            return {"recommendation": "Test recommendation."}

        with (
            patch("fpl_strategist.graph.fetch_context", stub_fetch),
            patch("fpl_strategist.graph.analyze_and_propose", stub_analyze),
            patch("fpl_strategist.graph.validate_constraints", stub_validate),
            patch("fpl_strategist.graph.select_captain", stub_captain),
            patch("fpl_strategist.graph.explain_recommendation", stub_explain),
        ):
            from fpl_strategist.graph import build_graph
            graph = build_graph()
            result = await graph.ainvoke({"team_id": 12345, "target_gw": 30})

        assert result["replan_count"] == 0
        assert result["is_valid"] is True
        assert result["proposed_transfer"] is not None
        assert result["proposed_transfer"]["out"]["id"] == 304   # Elanga
        assert result["proposed_transfer"]["in"]["id"] == 310    # Diaz

    @pytest.mark.asyncio
    async def test_single_replan_then_valid(self, sample_populated_state):
        """First proposal fails validation, replan proposes a different transfer that passes."""
        squad = sample_populated_state["current_squad"]
        candidates = sample_populated_state["candidates"]
        original_in_id = candidates[4]["id"]   # Diaz, id=310

        async def stub_fetch(state):
            return {
                "current_squad": squad, "bank": 25, "free_transfers": 1,
                "candidates": candidates, "fixtures": [],
                "teams": sample_populated_state["teams"], "player_form": {},
                "replan_count": 0, "proposed_transfer": None,
                "is_valid": False, "violations": [],
            }

        async def stub_analyze(state):
            return {
                "proposed_transfer": {"out": squad[11], "in": candidates[4]},
                "transfer_reasoning": "Elanga out, Diaz in.",
            }

        validate_calls = {"count": 0}

        async def stub_validate(state):
            validate_calls["count"] += 1
            if validate_calls["count"] == 1:
                return {"is_valid": False, "violations": ["Budget exceeded."]}
            return {"is_valid": True, "violations": []}

        async def stub_replan(state):
            return {
                "proposed_transfer": {"out": squad[11], "in": candidates[5]},
                "transfer_reasoning": "Revised: Elanga out, Gordon in.",
                "replan_count": state.get("replan_count", 0) + 1,
            }

        async def stub_captain(state):
            return {"captain_pick": {"player_id": 300}, "vice_captain_pick": {"player_id": 400}}

        async def stub_explain(state):
            return {"recommendation": "Test recommendation."}

        with (
            patch("fpl_strategist.graph.fetch_context", stub_fetch),
            patch("fpl_strategist.graph.analyze_and_propose", stub_analyze),
            patch("fpl_strategist.graph.validate_constraints", stub_validate),
            patch("fpl_strategist.graph.replan_transfer", stub_replan),
            patch("fpl_strategist.graph.select_captain", stub_captain),
            patch("fpl_strategist.graph.explain_recommendation", stub_explain),
        ):
            from fpl_strategist.graph import build_graph
            graph = build_graph()
            result = await graph.ainvoke({"team_id": 12345, "target_gw": 30})

        assert result["replan_count"] == 1
        assert result["is_valid"] is True
        assert result["proposed_transfer"] is not None
        assert result["proposed_transfer"]["in"]["id"] != original_in_id
        assert result["proposed_transfer"]["in"]["id"] == 311    # Gordon

    @pytest.mark.asyncio
    async def test_exhausted_replans_falls_back_to_hold(self, sample_populated_state):
        """Both replans fail — graph falls back to hold with cleared proposal."""
        squad = sample_populated_state["current_squad"]
        candidates = sample_populated_state["candidates"]

        async def stub_fetch(state):
            return {
                "current_squad": squad, "bank": 25, "free_transfers": 1,
                "candidates": candidates, "fixtures": [],
                "teams": sample_populated_state["teams"], "player_form": {},
                "replan_count": 0, "proposed_transfer": None,
                "is_valid": False, "violations": [],
            }

        async def stub_analyze(state):
            return {
                "proposed_transfer": {"out": squad[11], "in": candidates[4]},
                "transfer_reasoning": "Elanga out, Diaz in.",
            }

        async def stub_validate_always_fail(state):
            return {"is_valid": False, "violations": ["Always fails."]}

        async def stub_replan(state):
            return {
                "proposed_transfer": {"out": squad[11], "in": candidates[5]},
                "transfer_reasoning": "Revised but still invalid.",
                "replan_count": state.get("replan_count", 0) + 1,
            }

        async def stub_captain(state):
            return {"captain_pick": {"player_id": 300}, "vice_captain_pick": {"player_id": 400}}

        async def stub_explain(state):
            return {"recommendation": "Test recommendation."}

        with (
            patch("fpl_strategist.graph.fetch_context", stub_fetch),
            patch("fpl_strategist.graph.analyze_and_propose", stub_analyze),
            patch("fpl_strategist.graph.validate_constraints", stub_validate_always_fail),
            patch("fpl_strategist.graph.replan_transfer", stub_replan),
            patch("fpl_strategist.graph.select_captain", stub_captain),
            patch("fpl_strategist.graph.explain_recommendation", stub_explain),
        ):
            from fpl_strategist.graph import build_graph
            graph = build_graph()
            result = await graph.ainvoke({"team_id": 12345, "target_gw": 30})

        assert result["replan_count"] == 2
        assert result["is_valid"] is False
        assert result["proposed_transfer"] is None
        assert "hold" in result["transfer_reasoning"].lower()
