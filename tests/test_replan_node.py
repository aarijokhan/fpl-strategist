"""Tests for the replan_transfer LLM node."""

from __future__ import annotations

import copy

import pytest

from fpl_strategist.nodes.replan import replan_transfer


@pytest.mark.vcr("test_replan_avoids_violations.yaml")
async def test_replan_avoids_violations(sample_populated_state):
    """Seed state with a budget-busting proposal, invoke replan, verify correction."""
    state = copy.deepcopy(sample_populated_state)

    # Seed: Solanke (id=402, FWD, £7.5m) out → Isak (id=410, FWD, £9.0m) in
    # Bank = 10 (£1.0m). Available = 10 + 75 = 85, Isak costs 90 → budget violation.
    out_player = next(p for p in state["current_squad"] if p["id"] == 402)
    in_player = next(c for c in state["candidates"] if c["id"] == 410)

    state["bank"] = 10
    state["proposed_transfer"] = {"out": out_player, "in": in_player}
    state["transfer_reasoning"] = "Isak has great form and a decent fixture."
    state["violations"] = [
        "Insufficient budget: need £9.0m, have £8.5m (bank £1.0m + sale £7.5m)."
    ]
    state["replan_count"] = 0
    state["is_valid"] = False

    rejected_in_id = 410

    result = await replan_transfer(state)

    # replan_count must be incremented
    assert result["replan_count"] == 1

    # Reasoning is always a non-empty string
    assert isinstance(result["transfer_reasoning"], str)
    assert len(result["transfer_reasoning"]) > 0

    # If a transfer was proposed, it must differ from the rejected one
    if result["proposed_transfer"] is not None:
        new_in_id = result["proposed_transfer"]["in"]["id"]
        assert isinstance(new_in_id, int)
        assert new_in_id != rejected_in_id
