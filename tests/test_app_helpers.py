"""Unit tests for app_helpers — right-column display builders.

These are pure functions with no network calls, so no VCR/respx needed.
"""

from __future__ import annotations

import pytest

from app_helpers import (
    build_captain_card,
    build_meta_footer,
    build_squad_html,
    build_transfer_card,
)


def _squad_player(
    id, web_name, element_type, team, team_name, now_cost, form,
    fixture, *, status="a",
):
    pos_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    return {
        "id": id, "web_name": web_name, "element_type": element_type,
        "team": team, "team_name": team_name,
        "now_cost": now_cost, "form": form,
        "status": status, "position_name": pos_names[element_type],
        "fixture": fixture,
    }


def _make_state(*, with_transfer=True, with_captain=True):
    """Build a realistic state dict for helper tests."""
    fix_h = {"opponent": "SOU", "is_home": True, "difficulty": 2}
    fix_a = {"opponent": "MCI", "is_home": False, "difficulty": 4}
    fix_h2 = {"opponent": "BOU", "is_home": True, "difficulty": 2}

    squad = [
        _squad_player(1, "Alisson", 1, 12, "LIV", 55, "5.0", fix_h),
        _squad_player(2, "Raya", 1, 1, "ARS", 55, "4.5", fix_a),
        _squad_player(3, "Alexander-Arnold", 2, 12, "LIV", 85, "6.0", fix_h),
        _squad_player(4, "Saliba", 2, 1, "ARS", 60, "5.5", fix_a),
        _squad_player(5, "Gvardiol", 2, 13, "MCI", 60, "4.0", fix_a),
        _squad_player(6, "Estupinan", 2, 5, "BHA", 50, "3.0", fix_h2),
        _squad_player(7, "Mitchell", 2, 7, "CRY", 45, "2.0", fix_a),
        _squad_player(8, "Salah", 3, 12, "LIV", 130, "8.5", fix_h),
        _squad_player(9, "Palmer", 3, 6, "CHE", 110, "7.0", fix_h2),
        _squad_player(10, "Saka", 3, 1, "ARS", 100, "6.0", fix_a),
        _squad_player(11, "Mbeumo", 3, 4, "BRE", 75, "5.0", fix_h),
        _squad_player(12, "Elanga", 3, 16, "NFO", 55, "1.5", fix_a, status="d"),
        _squad_player(13, "Haaland", 4, 13, "MCI", 145, "7.0", fix_a),
        _squad_player(14, "Watkins", 4, 2, "AVL", 85, "5.5", fix_h),
        _squad_player(15, "Solanke", 4, 18, "TOT", 75, "4.0", fix_h2),
    ]

    state = {
        "team_id": 12345,
        "target_gw": 30,
        "current_squad": squad,
        "bank": 25,
        "free_transfers": 1,
        "replan_count": 1,
        "proposed_transfer": None,
        "transfer_reasoning": "",
        "captain_pick": None,
        "vice_captain_pick": None,
        "captain_reasoning": "",
    }

    if with_transfer:
        # Swap Elanga (id=12) for Diaz
        in_player = _squad_player(
            20, "Diaz", 3, 12, "LIV", 80, "6.5", fix_h,
        )
        state["proposed_transfer"] = {
            "out": squad[11],  # Elanga
            "in": in_player,
        }
        state["transfer_reasoning"] = "Elanga has poor form; Diaz offers better fixtures."

    if with_captain:
        state["captain_pick"] = {"id": 13, "name": "Haaland"}
        state["vice_captain_pick"] = {"id": 8, "name": "Salah"}
        state["captain_reasoning"] = "Haaland has great form and MCI face a weak side."

    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildSquadHtml:
    def test_structure(self):
        state = _make_state(with_transfer=True)
        html = build_squad_html(state)
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<div" in html

    def test_contains_all_player_names(self):
        state = _make_state(with_transfer=True)
        html = build_squad_html(state)
        # 14 original players (Elanga swapped out) + Diaz
        for name in [
            "Alisson", "Raya", "Alexander-Arnold", "Saliba", "Gvardiol",
            "Estupinan", "Mitchell", "Salah", "Palmer", "Saka", "Mbeumo",
            "Diaz", "Haaland", "Watkins", "Solanke",
        ]:
            assert name in html

    def test_transferred_in_player_has_prefix(self):
        state = _make_state(with_transfer=True)
        html = build_squad_html(state)
        assert "\U0001f504 Diaz" in html

    def test_no_transfer_no_prefix(self):
        state = _make_state(with_transfer=False)
        html = build_squad_html(state)
        assert "\U0001f504" not in html

    def test_status_emoji_mapping(self):
        state = _make_state(with_transfer=False)
        html = build_squad_html(state)
        # Elanga has status='d' -> 🟡
        assert "\U0001f7e1" in html
        # Available players -> 🟢
        assert "\U0001f7e2" in html


class TestBuildTransferCard:
    def test_with_transfer(self):
        state = _make_state(with_transfer=True)
        md = build_transfer_card(state)

        assert isinstance(md, str)
        assert len(md) > 0
        assert "Elanga" in md
        assert "Diaz" in md
        assert "### Transfer" in md
        assert "**OUT:**" in md
        assert "**IN:**" in md
        assert "**Net:**" in md

    def test_hold_case(self):
        state = _make_state(with_transfer=False)
        md = build_transfer_card(state)

        assert isinstance(md, str)
        assert "No transfer this gameweek" in md

    def test_hold_with_reasoning(self):
        state = _make_state(with_transfer=False)
        state["transfer_reasoning"] = "Squad is strong enough."
        md = build_transfer_card(state)

        assert "Squad is strong enough." in md


class TestBuildCaptainCard:
    def test_with_captain(self):
        state = _make_state(with_captain=True)
        md = build_captain_card(state)

        assert isinstance(md, str)
        assert len(md) > 0
        assert "Haaland" in md
        assert "Salah" in md
        assert "### Captaincy" in md
        assert "**Captain:**" in md
        assert "**Vice:**" in md

    def test_includes_reasoning(self):
        state = _make_state(with_captain=True)
        md = build_captain_card(state)

        assert "great form" in md


class TestBuildMetaFooter:
    def test_all_fields_present(self):
        state = _make_state(with_transfer=True)
        md = build_meta_footer(state)

        assert isinstance(md, str)
        assert len(md) > 0
        assert "GW 30" in md
        assert "\u00a32.5m" in md  # bank = 25 tenths = £2.5m
        assert "FT used: 1" in md
        assert "Replans: 1" in md

    def test_hold_ft_zero(self):
        state = _make_state(with_transfer=False)
        md = build_meta_footer(state)

        assert "FT used: 0" in md
