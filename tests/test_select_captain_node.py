"""Tests for the select_captain LLM node."""

from __future__ import annotations

import copy

import pytest

from fpl_strategist.nodes.select_captain import select_captain, _build_post_transfer_squad


@pytest.mark.vcr("test_select_captain_picks_captain.yaml")
async def test_select_captain_picks_captain(sample_populated_state):
    """Invoke select_captain with a valid post-transfer state and verify output."""
    state = copy.deepcopy(sample_populated_state)

    # Seed a valid transfer: Elanga (304, MID) out → Diaz (310, MID) in
    out_player = next(p for p in state["current_squad"] if p["id"] == 304)
    in_player = next(c for c in state["candidates"] if c["id"] == 310)
    state["proposed_transfer"] = {"out": out_player, "in": in_player}
    state["is_valid"] = True

    result = await select_captain(state)

    # Both picks must have id (int) and name (non-empty str)
    cap = result["captain_pick"]
    vc = result["vice_captain_pick"]

    assert isinstance(cap["id"], int)
    assert isinstance(cap["name"], str) and len(cap["name"]) > 0
    assert isinstance(vc["id"], int)
    assert isinstance(vc["name"], str) and len(vc["name"]) > 0

    # Captain and vice must differ
    assert cap["id"] != vc["id"]

    # Reasoning is non-empty
    assert isinstance(result["captain_reasoning"], str)
    assert len(result["captain_reasoning"]) > 0

    # Both IDs must be in the post-transfer squad
    post_squad = _build_post_transfer_squad(state)
    post_ids = {p["id"] for p in post_squad}
    assert cap["id"] in post_ids, f"Captain id={cap['id']} not in post-transfer squad"
    assert vc["id"] in post_ids, f"Vice-captain id={vc['id']} not in post-transfer squad"
