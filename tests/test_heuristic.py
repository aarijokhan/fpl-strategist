"""Tests for eval/heuristic.py — deterministic heuristic baseline."""

from __future__ import annotations

import pytest

from fpl_strategist.eval.heuristic import HeuristicResult, heuristic_recommend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(id, web_name, element_type, team, now_cost, form_float, difficulty=3, is_captain=False):
    """Build a minimal squad player dict."""
    return {
        "id": id,
        "web_name": web_name,
        "element_type": element_type,
        "team": team,
        "now_cost": now_cost,
        "selling_price": now_cost,
        "form": str(form_float),
        "form_float": form_float,
        "fixture": {"difficulty": difficulty},
    }


def _c(id, web_name, element_type, team, now_cost, form_float, difficulty=3):
    """Build a minimal candidate dict."""
    return {
        "id": id,
        "web_name": web_name,
        "element_type": element_type,
        "team": team,
        "now_cost": now_cost,
        "form_float": form_float,
        "fixture": {"difficulty": difficulty},
    }


def _base_squad():
    """15-player squad: 2 GKP, 5 DEF, 5 MID, 3 FWD. Elanga is the weak link."""
    return [
        # GKP (team 12, 13)
        _p(100, "Alisson",   1, 12, 55, 5.0),
        _p(101, "Raya",      1,  1, 55, 4.5),
        # DEF
        _p(200, "TAA",       2, 12, 85, 6.0),
        _p(201, "Saliba",    2,  1, 60, 5.5),
        _p(202, "Gvardiol",  2, 13, 60, 4.0),
        _p(203, "Pedro",     2, 18, 55, 3.0),
        _p(204, "Mitchell",  2,  7, 45, 2.0, difficulty=4),
        # MID
        _p(300, "Salah",     3, 12, 130, 8.5, difficulty=2),
        _p(301, "Palmer",    3,  6, 110, 7.0),
        _p(302, "Saka",      3,  1, 100, 6.0),
        _p(303, "Mbeumo",    3,  4, 75,  5.0),
        _p(304, "Elanga",    3, 16, 55,  1.5, difficulty=4),  # weakest outfield
        # FWD
        _p(400, "Haaland",   4, 13, 145, 7.0),
        _p(401, "Watkins",   4,  2, 85,  5.5),
        _p(402, "Solanke",   4, 18, 75,  4.0),
    ]


def _base_candidates():
    """A set of MID candidates for testing."""
    return [
        _c(310, "Diaz",   3, 12, 80,  6.5, difficulty=2),   # best
        _c(311, "Gordon", 3, 15, 75,  5.5, difficulty=3),
        _c(312, "Rogers", 3,  2, 55,  5.0, difficulty=2),
        _c(313, "Eze",    3,  7, 70,  4.5, difficulty=3),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPicksLowestCompositeOut:
    def test_weakest_outfield_selected(self):
        # Elanga: form 1.5 × (6-4) = 3.0, lowest among outfield
        result = heuristic_recommend(
            _base_squad(), _base_candidates(), bank=25,
            fixtures=[], teams={}, use_fixtures=True,
        )
        assert result.player_out is not None
        assert result.player_out["web_name"] == "Elanga"

    def test_weakest_form_only_mode(self):
        # Form-only: Elanga (1.5) still lowest
        result = heuristic_recommend(
            _base_squad(), _base_candidates(), bank=25,
            fixtures=[], teams={}, use_fixtures=False,
        )
        assert result.player_out is not None
        assert result.player_out["web_name"] == "Elanga"


class TestPicksHighestCompositeIn:
    def test_picks_best_affordable_candidate(self):
        result = heuristic_recommend(
            _base_squad(), _base_candidates(), bank=25,
            fixtures=[], teams={}, use_fixtures=True,
        )
        # Diaz (team 12/LIV) costs 80, exactly affordable — but _base_squad() already has
        # 3 LIV players (Alisson, TAA, Salah), so Diaz would create a 4th → rejected.
        # Next best is Rogers (team 2, 5.0 × (6-2) = 20.0), no team-limit issue.
        assert result.player_in is not None
        assert result.player_in["web_name"] == "Rogers"


class TestRespectsTeamLimit:
    def test_does_not_create_fourth_player_from_same_team(self):
        # Put 3 ARS players in squad (team 1), candidate is also ARS
        squad = _base_squad()
        # Make three existing players team 1 (Raya is already ARS)
        # Saliba (201) and Saka (302) are also team 1 — that's already 3.
        # Elanga (304) is the target for transfer out (team 16).
        # Rogers (312) is team 2, safe. Diaz (310) is team 12 (already has TAA+Alisson=2, ok).
        ars_candidates = [
            _c(320, "ARS_Player", 3, 1, 55, 9.0),  # would create 4th ARS player
            _c(311, "Gordon",     3, 15, 75, 5.5),  # safe
        ]
        result = heuristic_recommend(
            squad, ars_candidates, bank=25,
            fixtures=[], teams={},
        )
        # Should skip ARS_Player (would make 4 ARS) and pick Gordon instead
        assert result.player_in is not None
        assert result.player_in["web_name"] == "Gordon"

    def test_post_transfer_check_uses_updated_counts(self):
        # Out player is from team 1 (ARS), so after transfer team 1 goes from 3→2.
        # An ARS candidate should now be allowed.
        squad = [
            _p(100, "GKP",   1,  9, 45, 4.0),
            _p(101, "GKP2",  1, 10, 45, 4.0),
            _p(200, "DEF1",  2,  1, 55, 5.0),  # ARS
            _p(201, "DEF2",  2,  1, 55, 5.0),  # ARS
            _p(202, "DEF3",  2,  2, 55, 3.0),
            _p(203, "DEF4",  2,  3, 55, 3.0),
            _p(204, "DEF5",  2,  4, 45, 1.5),  # weakest outfield DEF
            _p(300, "MID1",  3,  5, 75, 5.0),
            _p(301, "MID2",  3,  6, 75, 5.0),
            _p(302, "MID3",  3,  7, 75, 5.0),
            _p(303, "MID4",  3,  8, 70, 5.0),
            _p(304, "MID5",  3,  1, 65, 4.0),  # ARS — 3rd ARS player
            _p(400, "FWD1",  4, 11, 85, 6.0),
            _p(401, "FWD2",  4, 12, 80, 5.5),
            _p(402, "FWD3",  4, 13, 75, 4.5),
        ]
        # Weakest DEF is DEF5 (team 4, form 1.5). Replacing it with an ARS DEF is
        # fine because: ARS currently has 3, but DEF5 is NOT ARS — we're checking
        # post-transfer for in-player's team. After removing DEF5 (team 4),
        # ARS count stays 3, so ARS DEF candidate is rejected.
        ars_def_candidate = _c(210, "ARS_DEF", 2, 1, 44, 6.0)
        safe_candidate    = _c(211, "Safe_DEF", 2, 9, 44, 5.5)
        result = heuristic_recommend(
            squad, [ars_def_candidate, safe_candidate], bank=0,
            fixtures=[], teams={},
        )
        # ARS already has 3 players, candidate would make 4 → rejected
        assert result.player_in is not None
        assert result.player_in["web_name"] == "Safe_DEF"


class TestRespectsTeamLimitBudget:
    def test_holds_when_all_candidates_too_expensive(self):
        expensive_candidates = [
            _c(310, "Expensive", 3, 15, 200, 8.0),
        ]
        result = heuristic_recommend(
            _base_squad(), expensive_candidates, bank=0,
            fixtures=[], teams={},
        )
        assert result.action == "hold"
        assert result.player_in is None


class TestCaptainSelection:
    def test_captain_is_highest_composite(self):
        result = heuristic_recommend(
            _base_squad(), _base_candidates(), bank=25,
            fixtures=[], teams={}, use_fixtures=True,
        )
        # Salah: form 8.5 × (6-2) = 34.0 — highest in squad
        assert result.captain_name == "Salah"

    def test_gkps_eligible_for_captaincy(self):
        # Make GKP have astronomical form
        squad = [
            _p(100, "SuperGKP", 1, 12, 55, 99.0, difficulty=1),
            _p(101, "Raya",     1,  1, 55,  4.5),
            _p(200, "DEF1",     2,  2, 55,  5.0),
            _p(201, "DEF2",     2,  3, 55,  4.0),
            _p(202, "DEF3",     2,  4, 55,  3.0),
            _p(203, "DEF4",     2,  5, 55,  3.0),
            _p(204, "DEF5",     2,  6, 45,  1.5),  # transfer-out candidate
            _p(300, "MID1",     3,  7, 75,  6.0),
            _p(301, "MID2",     3,  8, 75,  5.0),
            _p(302, "MID3",     3,  9, 75,  5.0),
            _p(303, "MID4",     3, 10, 70,  4.0),
            _p(304, "MID5",     3, 11, 65,  4.0),
            _p(400, "FWD1",     4, 13, 85,  7.0),
            _p(401, "FWD2",     4, 14, 80,  5.5),
            _p(402, "FWD3",     4, 15, 75,  4.5),
        ]
        candidates = [_c(210, "DEF_cand", 2, 16, 44, 5.0)]
        result = heuristic_recommend(squad, candidates, bank=1, fixtures=[], teams={})
        assert result.captain_name == "SuperGKP"


class TestExcludesGKPsFromTransferOut:
    def test_gkp_never_transfer_out(self):
        # Make GKP the lowest-form player
        squad = list(_base_squad())
        # Set both GKPs to very low form
        for p in squad:
            if p["element_type"] == 1:
                p["form_float"] = 0.0
                p["form"] = "0.0"
        result = heuristic_recommend(
            squad, _base_candidates(), bank=25,
            fixtures=[], teams={},
        )
        # Despite having the lowest form, GKPs must not be transfer-out target
        assert result.player_out is None or result.player_out.get("element_type") != 1


class TestFixtureModeVsFormOnly:
    def test_fixture_mode_can_differ_from_form_only(self):
        # Two MID players with different form/fixture trade-offs:
        # Player A: form 4.0, easy fixture (diff 1) → composite 4.0 × 5 = 20
        # Player B: form 5.0, hard fixture (diff 5) → composite 5.0 × 1 = 5
        # Form-only: B is lower (5.0 > 4.0 means B has HIGHER form, so A is lower-form transfer-out)
        # Wait, we want the WEAKEST player. Let me make it cleaner:
        # Player A: form 3.0, hard fixture (diff 5) → composite 3.0 × 1 = 3.0
        # Player B: form 4.0, easy fixture (diff 1) → composite 4.0 × 5 = 20.0
        # Player C: form 1.5, medium fixture (diff 3) → composite 1.5 × 3 = 4.5
        # Form-only: C (1.5) is weakest → transfer-out
        # Fixture mode: A (3.0) is weakest → transfer-out
        squad = [
            _p(100, "GKP",  1,  9, 45, 4.0),
            _p(101, "GKP2", 1, 10, 45, 4.0),
            _p(200, "DEF1", 2,  1, 55, 5.0),
            _p(201, "DEF2", 2,  2, 55, 5.0),
            _p(202, "DEF3", 2,  3, 55, 4.5),
            _p(203, "DEF4", 2,  4, 55, 4.5),
            _p(204, "DEF5", 2,  5, 45, 4.0),
            # MIDs with different form/fixture combos
            _p(300, "MidA", 3,  6, 75, 3.0, difficulty=5),  # composite=3.0
            _p(301, "MidB", 3,  7, 75, 4.0, difficulty=1),  # composite=20.0
            _p(302, "MidC", 3,  8, 70, 1.5, difficulty=3),  # form-only weakest
            _p(303, "MidD", 3, 11, 65, 5.0),
            _p(304, "MidE", 3, 12, 65, 5.5),
            _p(400, "FWD1", 4, 13, 85, 6.0),
            _p(401, "FWD2", 4, 14, 80, 5.5),
            _p(402, "FWD3", 4, 15, 75, 4.5),
        ]
        candidates = [_c(310, "MidIn", 3, 16, 65, 6.0, difficulty=2)]

        form_only  = heuristic_recommend(squad, candidates, bank=0, fixtures=[], teams={}, use_fixtures=False)
        fix_mode   = heuristic_recommend(squad, candidates, bank=0, fixtures=[], teams={}, use_fixtures=True)

        assert form_only.player_out is not None
        assert fix_mode.player_out is not None
        # Form-only: MidC (1.5) is weakest
        assert form_only.player_out["web_name"] == "MidC"
        # Fixture mode: MidA (composite 3.0) is weakest
        assert fix_mode.player_out["web_name"] == "MidA"
