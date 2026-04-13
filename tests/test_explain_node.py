"""Tests for the explain_recommendation LLM node."""

from __future__ import annotations

import copy

import pytest

from fpl_strategist.nodes.explain import explain_recommendation


@pytest.mark.vcr("test_explain_produces_recommendation.yaml")
async def test_explain_produces_recommendation(sample_populated_state):
    """Invoke explain_recommendation with a valid state and verify prose output."""
    state = copy.deepcopy(sample_populated_state)

    # Seed a valid transfer: Elanga (304, MID) out → Diaz (310, MID) in
    out_player = next(p for p in state["current_squad"] if p["id"] == 304)
    in_player = next(c for c in state["candidates"] if c["id"] == 310)
    state["proposed_transfer"] = {"out": out_player, "in": in_player}
    state["is_valid"] = True
    state["transfer_reasoning"] = (
        "Elanga has been in poor form (1.5) with a tough away fixture against "
        "Brighton (difficulty 3). Diaz is in strong form (6.5) with an easy "
        "home fixture against Southampton (difficulty 2) and EP next of 6.0."
    )

    # Captain picks
    state["captain_pick"] = {"id": 300, "name": "Salah"}
    state["vice_captain_pick"] = {"id": 400, "name": "Haaland"}
    state["captain_reasoning"] = (
        "Salah (form 8.5, EP 7.0) has a home fixture against Southampton "
        "(difficulty 2) — the easiest match of the week. Haaland (form 7.0) "
        "faces Arsenal away (difficulty 3), making him the ideal vice."
    )

    # No replan fired
    state["replan_count"] = 0

    result = await explain_recommendation(state)

    rec = result["recommendation"]

    # Must be substantial prose, not a stub
    assert isinstance(rec, str)
    assert len(rec) >= 500, f"Recommendation too short ({len(rec)} chars)"

    # Must mention at least one player from the transfer
    assert "Elanga" in rec or "Diaz" in rec, "Recommendation should mention a transfer player"

    # Must mention at least one captain pick
    assert "Salah" in rec or "Haaland" in rec, "Recommendation should mention a captain pick"
