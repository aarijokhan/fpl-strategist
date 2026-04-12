"""Tests for the validate_constraints graph node."""

from __future__ import annotations

import pytest

from fpl_strategist.nodes.validate import validate_constraints


def _player(
    id: int = 1,
    web_name: str = "Player",
    element_type: int = 3,
    team: int = 1,
    now_cost: int = 80,
    selling_price: int = 80,
    status: str = "a",
    news: str = "",
) -> dict:
    return {
        "id": id,
        "web_name": web_name,
        "element_type": element_type,
        "team": team,
        "now_cost": now_cost,
        "selling_price": selling_price,
        "status": status,
        "news": news,
    }


def _squad() -> list[dict]:
    return [
        _player(id=101, web_name="GKP1", element_type=1, team=1),
        _player(id=102, web_name="DEF1", element_type=2, team=2),
        _player(id=103, web_name="DEF2", element_type=2, team=3),
        _player(id=104, web_name="MID1", element_type=3, team=4),
        _player(id=105, web_name="FWD1", element_type=4, team=5),
    ]


TEAMS = {1: "ARS", 2: "AVL", 3: "BOU", 4: "BRE", 5: "BHA", 6: "CHE"}


@pytest.mark.asyncio
async def test_hold_transfer_is_valid():
    state = {
        "current_squad": _squad(),
        "bank": 15,
        "proposed_transfer": None,
        "teams": TEAMS,
    }
    result = await validate_constraints(state)
    assert result["is_valid"] is True
    assert result["violations"] == []


@pytest.mark.asyncio
async def test_valid_transfer():
    squad = _squad()
    out = squad[3]  # MID1, team=4
    inp = _player(id=200, web_name="NewMid", element_type=3, team=6, now_cost=75)
    state = {
        "current_squad": squad,
        "bank": 15,
        "proposed_transfer": {"out": out, "in": inp},
        "teams": TEAMS,
    }
    result = await validate_constraints(state)
    assert result["is_valid"] is True
    assert result["violations"] == []


@pytest.mark.asyncio
async def test_budget_violation():
    squad = _squad()
    out = squad[3]  # selling_price=80
    inp = _player(id=200, web_name="Expensive", element_type=3, team=6, now_cost=200)
    state = {
        "current_squad": squad,
        "bank": 15,
        "proposed_transfer": {"out": out, "in": inp},
        "teams": TEAMS,
    }
    result = await validate_constraints(state)
    assert result["is_valid"] is False
    assert any("Insufficient budget" in v for v in result["violations"])


@pytest.mark.asyncio
async def test_injured_player_violation():
    squad = _squad()
    out = squad[3]
    inp = _player(id=200, web_name="Injured", element_type=3, team=6, now_cost=50,
                  status="i", news="Knee")
    state = {
        "current_squad": squad,
        "bank": 15,
        "proposed_transfer": {"out": out, "in": inp},
        "teams": TEAMS,
    }
    result = await validate_constraints(state)
    assert result["is_valid"] is False
    assert any("injured" in v for v in result["violations"])


@pytest.mark.asyncio
async def test_position_mismatch_violation():
    squad = _squad()
    out = squad[3]  # MID
    inp = _player(id=200, web_name="NewDef", element_type=2, team=6, now_cost=50)
    state = {
        "current_squad": squad,
        "bank": 15,
        "proposed_transfer": {"out": out, "in": inp},
        "teams": TEAMS,
    }
    result = await validate_constraints(state)
    assert result["is_valid"] is False
    assert any("Position mismatch" in v for v in result["violations"])
