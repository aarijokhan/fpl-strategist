"""Tests for eval/historical.py — historical state reconstruction."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import respx
import httpx

from fpl_strategist.data.models import Fixture, Pick, PicksResponse
from fpl_strategist.eval.historical import (
    _compute_form,
    _historical_price,
    prefetch_all_histories,
    reconstruct_historical_state,
)


# ---------------------------------------------------------------------------
# Minimal test fixtures
# ---------------------------------------------------------------------------

def _make_picks_resp(
    player_ids: list[int],
    bank: int = 25,
    event_transfers: int = 0,
) -> PicksResponse:
    picks = [
        Pick(element=pid, position=i + 1, multiplier=1, is_captain=(i == 0), is_vice_captain=(i == 1))
        for i, pid in enumerate(player_ids)
    ]
    return PicksResponse(
        picks=picks,
        entry_history={"bank": bank, "event_transfers": event_transfers, "value": 1000},
    )


def _make_bootstrap(player_ids: list[int], team_id: int = 1) -> dict:
    elements = []
    for i, pid in enumerate(player_ids):
        elements.append({
            "id": pid,
            "web_name": f"Player{pid}",
            "element_type": 3,  # MID
            "team": team_id,
            "now_cost": 75,
            "form": "5.0",
            "ep_next": "5.0",
            "total_points": 100,
            "expected_goals": "0.50",
            "expected_assists": "0.30",
            "minutes": 1000,
            "news": "Hamstring",
            "chance_of_playing_next_round": 50,
            "status": "d",
        })
    return {
        "elements": elements,
        "teams": [{"id": team_id, "name": "Arsenal", "short_name": "ARS"}],
        "events": [],
    }


def _make_histories(
    player_ids: list[int],
    gw_range: range,
    pts_per_gw: int = 6,
    price: int = 75,
) -> dict[int, list[dict]]:
    histories = {}
    for pid in player_ids:
        histories[pid] = [
            {"round": gw, "total_points": pts_per_gw, "minutes": 90, "value": price}
            for gw in gw_range
        ]
    return histories


def _make_fixtures(target_gw: int = 15) -> list[Fixture]:
    return [
        Fixture(id=1, event=target_gw, team_h=1, team_a=2, team_h_difficulty=2, team_a_difficulty=4)
    ]


# ---------------------------------------------------------------------------
# _compute_form
# ---------------------------------------------------------------------------

class TestComputeForm:
    def test_form_computed_from_history(self):
        history = [{"round": gw, "total_points": gw} for gw in range(1, 21)]
        # target_gw=15 → window is rounds 10-14 → pts = 10+11+12+13+14=60 / 5 = 12.0
        result = _compute_form(history, target_gw=15)
        assert abs(result - 12.0) < 0.01

    def test_form_excludes_future_gws(self):
        # Rounds 14 (in window) and 15 (future — must be excluded)
        history = [
            {"round": 14, "total_points": 4},
            {"round": 15, "total_points": 99},  # must NOT be included
        ]
        result = _compute_form(history, target_gw=15)
        assert abs(result - 4.0) < 0.01

    def test_form_returns_zero_for_empty_history(self):
        assert _compute_form([], target_gw=15) == 0.0

    def test_form_ignores_rounds_before_window(self):
        history = [
            {"round": 1, "total_points": 99},   # outside 5-GW window
            {"round": 14, "total_points": 4},   # inside window
        ]
        result = _compute_form(history, target_gw=15)
        assert abs(result - 4.0) < 0.01


# ---------------------------------------------------------------------------
# _historical_price
# ---------------------------------------------------------------------------

class TestHistoricalPrice:
    def test_price_uses_historical_value(self):
        history = [
            {"round": 14, "value": 85},  # price at GW 14 = price before GW 15
            {"round": 15, "value": 90},  # future — should NOT be used
        ]
        result = _historical_price(history, target_gw=15, fallback=70)
        assert result == 85

    def test_falls_back_to_bootstrap_when_missing(self):
        result = _historical_price([], target_gw=15, fallback=70)
        assert result == 70


# ---------------------------------------------------------------------------
# reconstruct_historical_state
# ---------------------------------------------------------------------------

class TestReconstructHistoricalState:
    def test_all_players_set_available(self):
        """Status/news are not historically reconstructable — all players must be available."""
        player_ids = list(range(1, 16))
        bootstrap = _make_bootstrap(player_ids)
        histories = _make_histories(player_ids, range(1, 20))
        picks = _make_picks_resp(player_ids)
        fixtures = _make_fixtures()

        state = reconstruct_historical_state(
            team_id=44,
            target_gw=15,
            bootstrap_data=bootstrap,
            all_histories=histories,
            picks_resp=picks,
            fixtures=fixtures,
        )

        for player in state["current_squad"]:
            assert player["status"] == "a", f"{player['web_name']} should be available"
            assert player["news"] == ""

    def test_candidates_excludes_squad(self):
        """Candidates must not contain any current squad player IDs."""
        player_ids = list(range(1, 16))
        # Add extra non-squad players for the candidate pool
        all_player_ids = player_ids + list(range(100, 115))
        bootstrap = _make_bootstrap(all_player_ids)
        histories = _make_histories(all_player_ids, range(1, 20))
        picks = _make_picks_resp(player_ids)
        fixtures = _make_fixtures()

        state = reconstruct_historical_state(
            team_id=44,
            target_gw=15,
            bootstrap_data=bootstrap,
            all_histories=histories,
            picks_resp=picks,
            fixtures=fixtures,
        )

        squad_ids = {p["id"] for p in state["current_squad"]}
        candidate_ids = {c["id"] for c in state["candidates"]}
        overlap = squad_ids & candidate_ids
        assert not overlap, f"Candidates include squad players: {overlap}"

    def test_state_shape_matches_fetch_context(self):
        """All keys returned by fetch_context must be present."""
        required_keys = {
            "current_squad", "bank", "free_transfers", "candidates",
            "fixtures", "teams", "player_form", "replan_count",
            "proposed_transfer", "is_valid", "violations",
        }
        player_ids = list(range(1, 16))
        bootstrap = _make_bootstrap(player_ids)
        histories = _make_histories(player_ids, range(1, 20))
        picks = _make_picks_resp(player_ids)
        fixtures = _make_fixtures()

        state = reconstruct_historical_state(
            team_id=44,
            target_gw=15,
            bootstrap_data=bootstrap,
            all_histories=histories,
            picks_resp=picks,
            fixtures=fixtures,
        )
        assert required_keys <= set(state.keys())
        assert state["replan_count"] == 0
        assert state["proposed_transfer"] is None
        assert state["is_valid"] is False

    def test_free_transfers_from_prior_gw(self):
        """free_transfers uses event_transfers from GW N-1 picks, not target GW data."""
        player_ids = list(range(1, 16))
        bootstrap = _make_bootstrap(player_ids)
        histories = _make_histories(player_ids, range(1, 20))
        # event_transfers=1 means manager used their 1 free transfer in GW N-1
        # → they get 1 free transfer for GW N (2 - 1 = 1, but min is 1 anyway)
        picks = _make_picks_resp(player_ids, event_transfers=1)
        fixtures = _make_fixtures()

        state = reconstruct_historical_state(
            team_id=44,
            target_gw=15,
            bootstrap_data=bootstrap,
            all_histories=histories,
            picks_resp=picks,
            fixtures=fixtures,
        )
        assert state["free_transfers"] == 1

    def test_free_transfers_rollover(self):
        """event_transfers=0 in prior GW means the manager has 2 free transfers."""
        player_ids = list(range(1, 16))
        bootstrap = _make_bootstrap(player_ids)
        histories = _make_histories(player_ids, range(1, 20))
        picks = _make_picks_resp(player_ids, event_transfers=0)
        fixtures = _make_fixtures()

        state = reconstruct_historical_state(
            team_id=44,
            target_gw=15,
            bootstrap_data=bootstrap,
            all_histories=histories,
            picks_resp=picks,
            fixtures=fixtures,
        )
        assert state["free_transfers"] == 2


# ---------------------------------------------------------------------------
# prefetch_all_histories
# ---------------------------------------------------------------------------

class TestPrefetchAllHistories:
    async def test_returns_histories_for_all_players(self):
        """Verify prefetch returns a dict for each player ID."""
        from fpl_strategist.data.fpl_client import FPLClient

        player_ids = [1, 2, 3]
        mock_response = {"history": [{"round": 10, "total_points": 5, "minutes": 90, "value": 75}]}

        with respx.mock:
            for pid in player_ids:
                respx.get(f"https://fantasy.premierleague.com/api/element-summary/{pid}/").mock(
                    return_value=httpx.Response(200, json=mock_response)
                )
            async with FPLClient() as client:
                semaphore = asyncio.Semaphore(5)
                result = await prefetch_all_histories(client, player_ids, semaphore)

        assert set(result.keys()) == set(player_ids)
        for pid in player_ids:
            assert result[pid] == mock_response["history"]

    async def test_prefetch_respects_semaphore(self):
        """With semaphore=2, no more than 2 concurrent requests should be active."""
        from fpl_strategist.data.fpl_client import FPLClient

        player_ids = list(range(1, 6))
        active_count = 0
        max_concurrent = 0

        async def mock_handler(request):
            nonlocal active_count, max_concurrent
            active_count += 1
            max_concurrent = max(max_concurrent, active_count)
            await asyncio.sleep(0.01)
            active_count -= 1
            return httpx.Response(200, json={"history": []})

        with respx.mock:
            for pid in player_ids:
                respx.get(f"https://fantasy.premierleague.com/api/element-summary/{pid}/").mock(
                    side_effect=mock_handler
                )
            async with FPLClient() as client:
                semaphore = asyncio.Semaphore(2)
                await prefetch_all_histories(client, player_ids, semaphore)

        assert max_concurrent <= 2

    async def test_failed_fetch_skipped_gracefully(self):
        """A 500 error for one player should not crash the whole prefetch."""
        from fpl_strategist.data.fpl_client import FPLClient

        with respx.mock:
            respx.get("https://fantasy.premierleague.com/api/element-summary/1/").mock(
                return_value=httpx.Response(200, json={"history": [{"round": 10, "total_points": 5}]})
            )
            respx.get("https://fantasy.premierleague.com/api/element-summary/2/").mock(
                return_value=httpx.Response(500)
            )
            async with FPLClient() as client:
                semaphore = asyncio.Semaphore(5)
                result = await prefetch_all_histories(client, [1, 2], semaphore)

        # Player 1 should succeed, player 2 should return empty list
        assert 1 in result
        assert result[2] == []
