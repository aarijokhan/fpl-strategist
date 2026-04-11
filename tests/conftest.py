"""Shared test fixtures for FPL Strategist tests."""

from __future__ import annotations

import pytest

from fpl_strategist.data.models import Fixture, Player


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
