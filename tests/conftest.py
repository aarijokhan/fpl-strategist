"""Shared test fixtures for FPL Strategist tests."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from fpl_strategist.data.models import Fixture, Player

# Load .env so API keys are available during VCR recording.
# In playback mode the actual key isn't used (VCR replays from cassette),
# but ChatOpenAI still validates that the env var exists at construction time.
# CI has no .env — set a dummy key so the constructor doesn't blow up.
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-test-placeholder-for-vcr-playback"


# ---------------------------------------------------------------------------
# VCR / pytest-recording configuration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "openai-organization"],
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return os.path.join(os.path.dirname(__file__), "cassettes")


def _make_player(
    id: int = 1,
    web_name: str = "TestPlayer",
    element_type: int = 3,
    team: int = 1,
    now_cost: int = 80,
    form: str = "5.0",
    total_points: int = 100,
    ep_next: str | None = "4.5",
    status: str = "a",
    news: str = "",
    chance_of_playing_next_round: int | None = 100,
    **kwargs,
) -> Player:
    return Player(
        id=id,
        web_name=web_name,
        element_type=element_type,
        team=team,
        now_cost=now_cost,
        form=form,
        total_points=total_points,
        ep_next=ep_next,
        status=status,
        news=news,
        chance_of_playing_next_round=chance_of_playing_next_round,
        **kwargs,
    )


def _make_fixture(
    id: int = 1,
    event: int = 10,
    team_h: int = 1,
    team_a: int = 2,
    team_h_difficulty: int = 2,
    team_a_difficulty: int = 3,
) -> Fixture:
    return Fixture(
        id=id,
        event=event,
        team_h=team_h,
        team_a=team_a,
        team_h_difficulty=team_h_difficulty,
        team_a_difficulty=team_a_difficulty,
    )


@pytest.fixture
def make_player():
    return _make_player


@pytest.fixture
def make_fixture():
    return _make_fixture


@pytest.fixture
def sample_bootstrap_response() -> dict:
    """Minimal bootstrap-static response for testing."""
    return {
        "elements": [
            {
                "id": 1, "web_name": "Salah", "element_type": 3, "team": 10,
                "now_cost": 130, "form": "8.5", "total_points": 200,
                "ep_next": "7.0", "expected_goals": "0.80", "expected_assists": "0.50",
                "minutes": 2700, "news": "", "chance_of_playing_next_round": 100,
                "status": "a",
            },
            {
                "id": 2, "web_name": "Haaland", "element_type": 4, "team": 11,
                "now_cost": 145, "form": "7.2", "total_points": 180,
                "ep_next": "6.5", "expected_goals": "1.00", "expected_assists": "0.20",
                "minutes": 2500, "news": "", "chance_of_playing_next_round": 100,
                "status": "a",
            },
            {
                "id": 3, "web_name": "Saka", "element_type": 3, "team": 1,
                "now_cost": 105, "form": "6.0", "total_points": 160,
                "ep_next": "5.5", "expected_goals": "0.40", "expected_assists": "0.60",
                "minutes": 2600, "news": "", "chance_of_playing_next_round": 100,
                "status": "a",
            },
            {
                "id": 4, "web_name": "Injured", "element_type": 2, "team": 5,
                "now_cost": 55, "form": "0.0", "total_points": 50,
                "ep_next": "0.0", "expected_goals": "0.00", "expected_assists": "0.00",
                "minutes": 500, "news": "Knee injury", "chance_of_playing_next_round": 0,
                "status": "i",
            },
        ],
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 5, "name": "Chelsea", "short_name": "CHE"},
            {"id": 10, "name": "Liverpool", "short_name": "LIV"},
            {"id": 11, "name": "Man City", "short_name": "MCI"},
        ],
        "events": [
            {"id": 9, "name": "Gameweek 9", "deadline_time": "2025-10-25T11:00:00Z",
             "is_current": True, "is_next": False, "finished": False},
            {"id": 10, "name": "Gameweek 10", "deadline_time": "2025-11-01T11:00:00Z",
             "is_current": False, "is_next": True, "finished": False},
        ],
    }


@pytest.fixture
def sample_entry_response() -> dict:
    return {
        "id": 12345,
        "player_first_name": "Test",
        "player_last_name": "Manager",
        "name": "Test FC",
        "summary_overall_points": 500,
        "summary_overall_rank": 100000,
    }


@pytest.fixture
def sample_picks_response() -> dict:
    return {
        "picks": [
            {"element": 1, "position": 1, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
            {"element": 2, "position": 2, "multiplier": 1, "is_captain": False, "is_vice_captain": True},
            {"element": 3, "position": 3, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        ],
        "entry_history": {
            "bank": 15,
            "value": 1000,
            "event_transfers": 0,
            "event_transfers_cost": 0,
            "points": 60,
            "total_points": 500,
        },
    }


@pytest.fixture
def sample_fixtures_response() -> list[dict]:
    return [
        {
            "id": 1, "event": 10,
            "team_h": 10, "team_a": 5,
            "team_h_difficulty": 2, "team_a_difficulty": 4,
            "finished": False, "team_h_score": None, "team_a_score": None,
        },
        {
            "id": 2, "event": 10,
            "team_h": 1, "team_a": 11,
            "team_h_difficulty": 4, "team_a_difficulty": 3,
            "finished": False, "team_h_score": None, "team_a_score": None,
        },
    ]


# ---------------------------------------------------------------------------
# Populated state for LLM node tests
# ---------------------------------------------------------------------------

def _squad_player(
    id, web_name, element_type, team, team_name, now_cost, form,
    ep_next, total_points, fixture, *, status="a", news="",
    is_captain=False, is_vice_captain=False, multiplier=1,
):
    """Helper to build a squad player dict matching fetch_context output shape."""
    pos_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    return {
        "id": id, "web_name": web_name, "element_type": element_type,
        "team": team, "team_name": team_name,
        "now_cost": now_cost, "selling_price": now_cost,
        "form": form, "form_float": float(form),
        "total_points": total_points,
        "ep_next": ep_next, "ep_next_float": float(ep_next) if ep_next else 0.0,
        "expected_goals": "0.00", "expected_assists": "0.00",
        "minutes": 2000, "news": news, "status": status,
        "chance_of_playing_next_round": 100,
        "position_name": pos_names.get(element_type, "UNK"),
        "is_captain": is_captain, "is_vice_captain": is_vice_captain,
        "multiplier": multiplier, "fixture": fixture,
    }


def _candidate(id, web_name, element_type, team, team_name, now_cost, form, ep_next, fixture, *, status="a", news=""):
    pos_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    return {
        "id": id, "web_name": web_name, "element_type": element_type,
        "team": team, "team_name": team_name,
        "now_cost": now_cost, "form": form, "form_float": float(form),
        "ep_next": ep_next, "ep_next_float": float(ep_next) if ep_next else 0.0,
        "total_points": 100, "status": status, "news": news,
        "position_name": pos_names.get(element_type, "UNK"),
        "fixture": fixture,
    }


@pytest.fixture
def sample_populated_state() -> dict:
    """A realistic FPLState for LLM node tests.

    15-player squad with two clear weak links (Elanga form=1.5, Mitchell form=2.0),
    a healthy bank, and ~11 transfer candidates across all positions.
    """
    # Fixture info dicts (opponent, is_home, difficulty)
    fix_liv = {"opponent": "SOU", "is_home": True, "difficulty": 2}
    fix_ars = {"opponent": "MCI", "is_home": True, "difficulty": 4}
    fix_mci = {"opponent": "ARS", "is_home": False, "difficulty": 3}
    fix_bha = {"opponent": "NFO", "is_home": True, "difficulty": 2}
    fix_cry = {"opponent": "BRE", "is_home": False, "difficulty": 3}
    fix_che = {"opponent": "BOU", "is_home": True, "difficulty": 2}
    fix_bre = {"opponent": "CRY", "is_home": True, "difficulty": 2}
    fix_nfo = {"opponent": "BHA", "is_home": False, "difficulty": 3}
    fix_avl = {"opponent": "NEW", "is_home": True, "difficulty": 3}
    fix_tot = {"opponent": "EVE", "is_home": True, "difficulty": 2}
    fix_new = {"opponent": "AVL", "is_home": False, "difficulty": 3}
    fix_eve = {"opponent": "TOT", "is_home": False, "difficulty": 4}

    squad = [
        # GKP
        _squad_player(100, "Alisson", 1, 12, "LIV", 55, "5.0", "4.5", 120, fix_liv),
        _squad_player(101, "Raya", 1, 1, "ARS", 55, "4.5", "4.0", 115, fix_ars, multiplier=0),
        # DEF
        _squad_player(200, "Alexander-Arnold", 2, 12, "LIV", 85, "6.0", "5.5", 140, fix_liv),
        _squad_player(201, "Saliba", 2, 1, "ARS", 60, "5.5", "4.5", 130, fix_ars),
        _squad_player(202, "Gvardiol", 2, 13, "MCI", 60, "4.0", "3.5", 100, fix_mci),
        _squad_player(203, "Estupinan", 2, 5, "BHA", 50, "3.0", "2.5", 70, fix_bha),
        _squad_player(204, "Mitchell", 2, 7, "CRY", 45, "2.0", "1.5", 55, fix_cry, multiplier=0),
        # MID
        _squad_player(300, "Salah", 3, 12, "LIV", 130, "8.5", "7.0", 200, fix_liv, is_captain=True, multiplier=2),
        _squad_player(301, "Palmer", 3, 6, "CHE", 110, "7.0", "6.0", 175, fix_che),
        _squad_player(302, "Saka", 3, 1, "ARS", 100, "6.0", "5.5", 160, fix_ars),
        _squad_player(303, "Mbeumo", 3, 4, "BRE", 75, "5.0", "4.5", 120, fix_bre),
        _squad_player(304, "Elanga", 3, 16, "NFO", 55, "1.5", "1.0", 45, fix_nfo),
        # FWD
        _squad_player(400, "Haaland", 4, 13, "MCI", 145, "7.0", "6.5", 180, fix_mci, is_vice_captain=True),
        _squad_player(401, "Watkins", 4, 2, "AVL", 85, "5.5", "5.0", 130, fix_avl),
        _squad_player(402, "Solanke", 4, 18, "TOT", 75, "4.0", "3.5", 90, fix_tot),
    ]

    candidates = [
        # GKP
        _candidate(110, "Flekken", 1, 4, "BRE", 45, "4.0", "3.5", fix_bre),
        # DEF
        _candidate(210, "Pedro Porro", 2, 18, "TOT", 55, "6.0", "5.0", fix_tot),
        _candidate(211, "Gabriel", 2, 1, "ARS", 55, "5.0", "4.0", fix_ars),
        _candidate(212, "Mykolenko", 2, 8, "EVE", 44, "4.0", "3.0", fix_eve),
        # MID
        _candidate(310, "Diaz", 3, 12, "LIV", 80, "6.5", "6.0", fix_liv),
        _candidate(311, "Gordon", 3, 15, "NEW", 75, "5.5", "5.0", fix_new),
        _candidate(312, "Rogers", 3, 2, "AVL", 55, "5.0", "4.5", fix_avl),
        _candidate(313, "Eze", 3, 7, "CRY", 70, "4.5", "4.0", fix_cry),
        # FWD
        _candidate(410, "Isak", 4, 15, "NEW", 90, "7.5", "7.0", fix_new),
        _candidate(412, "Joao Pedro", 4, 5, "BHA", 60, "5.0", "4.5", fix_bha),
        _candidate(413, "Welbeck", 4, 5, "BHA", 55, "4.5", "4.0", fix_bha),
    ]

    teams = {
        1: "ARS", 2: "AVL", 4: "BRE", 5: "BHA", 6: "CHE", 7: "CRY",
        8: "EVE", 12: "LIV", 13: "MCI", 15: "NEW", 16: "NFO", 18: "TOT",
    }

    return {
        "team_id": 12345,
        "target_gw": 30,
        "provider": "openai",
        "current_squad": squad,
        "bank": 25,  # £2.5m
        "free_transfers": 1,
        "candidates": candidates,
        "fixtures": [],  # raw fixture dicts not needed by analyze node
        "teams": teams,
        "player_form": {},
        "replan_count": 0,
        "proposed_transfer": None,
        "is_valid": False,
        "violations": [],
    }
