"""Thorough unit tests for the constraint validation engine."""

from __future__ import annotations

import pytest

from fpl_strategist.constraints.engine import validate_transfer, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player(
    id: int = 1,
    web_name: str = "Player",
    element_type: int = 3,
    team: int = 1,
    now_cost: int = 80,
    selling_price: int | None = None,
    status: str = "a",
    news: str = "",
) -> dict:
    d = {
        "id": id,
        "web_name": web_name,
        "element_type": element_type,
        "team": team,
        "now_cost": now_cost,
        "status": status,
        "news": news,
    }
    if selling_price is not None:
        d["selling_price"] = selling_price
    return d


def _squad(*overrides: dict) -> list[dict]:
    """Build a default 15-player squad. Override specific slots with dicts."""
    defaults = [
        _player(id=101, web_name="GKP1", element_type=1, team=1, now_cost=45, selling_price=45),
        _player(id=102, web_name="GKP2", element_type=1, team=2, now_cost=40, selling_price=40),
        _player(id=103, web_name="DEF1", element_type=2, team=3, now_cost=55, selling_price=55),
        _player(id=104, web_name="DEF2", element_type=2, team=4, now_cost=50, selling_price=50),
        _player(id=105, web_name="DEF3", element_type=2, team=5, now_cost=50, selling_price=50),
        _player(id=106, web_name="DEF4", element_type=2, team=6, now_cost=45, selling_price=45),
        _player(id=107, web_name="DEF5", element_type=2, team=7, now_cost=45, selling_price=45),
        _player(id=108, web_name="MID1", element_type=3, team=8, now_cost=80, selling_price=80),
        _player(id=109, web_name="MID2", element_type=3, team=9, now_cost=75, selling_price=75),
        _player(id=110, web_name="MID3", element_type=3, team=10, now_cost=70, selling_price=70),
        _player(id=111, web_name="MID4", element_type=3, team=11, now_cost=65, selling_price=65),
        _player(id=112, web_name="MID5", element_type=3, team=12, now_cost=60, selling_price=60),
        _player(id=113, web_name="FWD1", element_type=4, team=13, now_cost=100, selling_price=100),
        _player(id=114, web_name="FWD2", element_type=4, team=14, now_cost=90, selling_price=90),
        _player(id=115, web_name="FWD3", element_type=4, team=15, now_cost=85, selling_price=85),
    ]
    squad = list(defaults)
    for o in overrides:
        for i, p in enumerate(squad):
            if p["id"] == o["id"]:
                squad[i] = {**p, **o}
                break
    return squad


TEAMS = {
    1: "ARS", 2: "AVL", 3: "BOU", 4: "BRE", 5: "BHA",
    6: "CHE", 7: "CRY", 8: "EVE", 9: "FUL", 10: "LIV",
    11: "MCI", 12: "MUN", 13: "NEW", 14: "NFO", 15: "TOT",
    16: "WHU", 17: "WOL", 18: "IPS", 19: "LEI", 20: "SOU",
}


# ---------------------------------------------------------------------------
# Hold transfer
# ---------------------------------------------------------------------------

class TestHoldTransfer:
    def test_hold_is_always_valid(self):
        result = validate_transfer(_squad(), bank=15, proposed_out=None, proposed_in=None)
        assert result.is_valid is True
        assert result.violations == []


# ---------------------------------------------------------------------------
# Valid transfers
# ---------------------------------------------------------------------------

class TestValidTransfer:
    def test_basic_valid_transfer(self):
        squad = _squad()
        out = squad[7]  # MID1, team=8, cost=80
        inp = _player(id=200, web_name="NewMid", element_type=3, team=16, now_cost=75)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp, teams=TEAMS)
        assert result.is_valid is True
        assert result.violations == []

    def test_exact_budget_transfer(self):
        squad = _squad()
        out = squad[7]  # selling_price=80
        # Can afford exactly: bank(15) + sell(80) = 95
        inp = _player(id=200, web_name="ExactMid", element_type=3, team=16, now_cost=95)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert result.is_valid is True

    def test_doubtful_player_is_allowed(self):
        squad = _squad()
        out = squad[7]
        inp = _player(id=200, web_name="DoubtfulMid", element_type=3, team=16, now_cost=75, status="d")
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Rule 1: No self-swap
# ---------------------------------------------------------------------------

class TestNoSelfSwap:
    def test_self_swap_rejected(self):
        squad = _squad()
        player = squad[7]
        result = validate_transfer(squad, bank=15, proposed_out=player, proposed_in=player)
        assert result.is_valid is False
        assert any("for themselves" in v for v in result.violations)


# ---------------------------------------------------------------------------
# Rule 2: Squad membership
# ---------------------------------------------------------------------------

class TestSquadMembership:
    def test_out_player_not_in_squad(self):
        squad = _squad()
        out = _player(id=999, web_name="NotInSquad", element_type=3, team=16, now_cost=50)
        inp = _player(id=200, web_name="NewMid", element_type=3, team=17, now_cost=50)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert result.is_valid is False
        assert any("not in your current squad" in v for v in result.violations)


# ---------------------------------------------------------------------------
# Rule 3: Player existence
# ---------------------------------------------------------------------------

class TestPlayerExistence:
    def test_in_player_not_in_pool(self):
        squad = _squad()
        out = squad[7]
        inp = _player(id=999, web_name="Ghost", element_type=3, team=16, now_cost=50)
        pool = [_player(id=i, web_name=f"P{i}", element_type=3, team=i) for i in range(1, 5)]
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp, all_players=pool)
        assert result.is_valid is False
        assert any("not found in player pool" in v for v in result.violations)

    def test_pool_not_provided_skips_check(self):
        squad = _squad()
        out = squad[7]
        inp = _player(id=999, web_name="AnyPlayer", element_type=3, team=16, now_cost=50)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp, all_players=None)
        # Should not fail on pool check when pool is not provided
        assert not any("not found in player pool" in v for v in result.violations)


# ---------------------------------------------------------------------------
# Rule 4: Position match
# ---------------------------------------------------------------------------

class TestPositionMatch:
    def test_position_mismatch_rejected(self):
        squad = _squad()
        out = squad[7]  # MID (element_type=3)
        inp = _player(id=200, web_name="NewDef", element_type=2, team=16, now_cost=50)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert result.is_valid is False
        assert any("Position mismatch" in v for v in result.violations)
        assert any("MID" in v and "DEF" in v for v in result.violations)

    def test_same_position_passes(self):
        squad = _squad()
        out = squad[2]  # DEF1 (element_type=2)
        inp = _player(id=200, web_name="NewDef", element_type=2, team=16, now_cost=50)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert not any("Position mismatch" in v for v in result.violations)


# ---------------------------------------------------------------------------
# Rule 5: Budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_over_budget_rejected(self):
        squad = _squad()
        out = squad[7]  # selling_price=80
        # Need 200, have bank(15) + sell(80) = 95
        inp = _player(id=200, web_name="Expensive", element_type=3, team=16, now_cost=200)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp, teams=TEAMS)
        assert result.is_valid is False
        assert any("Insufficient budget" in v for v in result.violations)

    def test_one_over_budget_rejected(self):
        squad = _squad()
        out = squad[7]  # selling_price=80
        # Need 96, have 95
        inp = _player(id=200, web_name="JustOver", element_type=3, team=16, now_cost=96)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert result.is_valid is False

    def test_zero_bank_uses_selling_price(self):
        squad = _squad()
        out = squad[7]  # selling_price=80
        inp = _player(id=200, web_name="Affordable", element_type=3, team=16, now_cost=80)
        result = validate_transfer(squad, bank=0, proposed_out=out, proposed_in=inp)
        assert result.is_valid is True

    def test_uses_selling_price_over_now_cost(self):
        """selling_price should be used for the outgoing player, not now_cost."""
        squad = _squad({"id": 108, "selling_price": 70, "now_cost": 80})
        out = squad[7]  # selling_price=70 (less than now_cost=80)
        # bank(0) + sell(70) = 70, need 75
        inp = _player(id=200, web_name="Mid", element_type=3, team=16, now_cost=75)
        result = validate_transfer(squad, bank=0, proposed_out=out, proposed_in=inp)
        assert result.is_valid is False


# ---------------------------------------------------------------------------
# Rule 6: Team limit
# ---------------------------------------------------------------------------

class TestTeamLimit:
    def test_team_limit_exceeded(self):
        # Put 3 players on team 16, then try to add a 4th
        squad = _squad(
            {"id": 108, "team": 16},
            {"id": 109, "team": 16},
            {"id": 110, "team": 16},
        )
        out = squad[11]  # MID4, team=11 -> removing from team 11
        inp = _player(id=200, web_name="FourthOnTeam", element_type=3, team=16, now_cost=50)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp, teams=TEAMS)
        assert result.is_valid is False
        assert any("Team limit exceeded" in v for v in result.violations)
        assert any("4 players" in v for v in result.violations)

    def test_replacing_same_team_stays_within_limit(self):
        # 3 on team 16, replacing one of them -> still 3
        squad = _squad(
            {"id": 108, "team": 16},
            {"id": 109, "team": 16},
            {"id": 110, "team": 16},
        )
        out = squad[7]  # MID1, now on team 16
        inp = _player(id=200, web_name="SameTeamReplacement", element_type=3, team=16, now_cost=50)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp, teams=TEAMS)
        assert not any("Team limit" in v for v in result.violations)

    def test_team_name_in_violation_message(self):
        squad = _squad(
            {"id": 108, "team": 11},
            {"id": 109, "team": 11},
            {"id": 110, "team": 11},
        )
        out = squad[11]  # MID4, also team=11 originally, keeping 3 of team 11
        # Actually out is MID4 which is team=11. After removing MID4, team 11 has 3 - 1 = 2.
        # Let me fix: transfer out MID5 (team=12), transfer in another team 11 player
        out = squad[11]  # MID4 (team 11)
        # After removing MID4 (team 11): team 11 has 2 (MID1=108, MID2=109, MID3=110 minus MID4=111... wait
        # Let me restructure: 108,109,110 are on team 11. Out=112 (team 12). In=team 11.
        out = squad[12]  # FWD1, team=13
        inp = _player(id=200, web_name="NewFwd", element_type=4, team=11, now_cost=50)
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp, teams=TEAMS)
        assert result.is_valid is False
        assert any("MCI" in v for v in result.violations)


# ---------------------------------------------------------------------------
# Rule 7: Availability
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_injured_player_rejected(self):
        squad = _squad()
        out = squad[7]
        inp = _player(id=200, web_name="InjuredGuy", element_type=3, team=16,
                      now_cost=50, status="i", news="Hamstring strain")
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert result.is_valid is False
        assert any("injured" in v for v in result.violations)
        assert any("Hamstring strain" in v for v in result.violations)

    def test_suspended_player_rejected(self):
        squad = _squad()
        out = squad[7]
        inp = _player(id=200, web_name="RedCarded", element_type=3, team=16,
                      now_cost=50, status="s")
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert result.is_valid is False
        assert any("suspended" in v for v in result.violations)

    def test_unavailable_player_rejected(self):
        squad = _squad()
        out = squad[7]
        inp = _player(id=200, web_name="Gone", element_type=3, team=16,
                      now_cost=50, status="u")
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert result.is_valid is False
        assert any("unavailable" in v for v in result.violations)

    def test_available_player_passes(self):
        squad = _squad()
        out = squad[7]
        inp = _player(id=200, web_name="Fit", element_type=3, team=16, now_cost=50, status="a")
        result = validate_transfer(squad, bank=15, proposed_out=out, proposed_in=inp)
        assert not any("injured" in v or "suspended" in v or "unavailable" in v for v in result.violations)


# ---------------------------------------------------------------------------
# Multiple violations
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    def test_multiple_violations_collected(self):
        """A transfer that violates budget, position, AND availability should report all."""
        squad = _squad()
        out = squad[7]  # MID, selling_price=80
        inp = _player(
            id=200, web_name="BadChoice", element_type=2,  # DEF != MID
            team=16, now_cost=200,  # way over budget
            status="i", news="ACL",  # injured
        )
        result = validate_transfer(squad, bank=0, proposed_out=out, proposed_in=inp)
        assert result.is_valid is False
        assert len(result.violations) >= 3
        types = " ".join(result.violations)
        assert "Position mismatch" in types
        assert "Insufficient budget" in types
        assert "injured" in types

    def test_partial_transfer_rejected(self):
        """Providing only out or only in is invalid."""
        squad = _squad()
        result = validate_transfer(squad, bank=15, proposed_out=squad[0], proposed_in=None)
        assert result.is_valid is False
        assert any("both" in v.lower() for v in result.violations)
