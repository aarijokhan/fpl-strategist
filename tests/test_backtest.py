"""Tests for eval/backtest.py — backtest harness."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from fpl_strategist.eval.backtest import build_backtest_graph, run_single_gw
from fpl_strategist.eval.scoring import GWResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_picks_resp(player_ids: list[int] = None, event_transfers: int = 0):
    from fpl_strategist.data.models import Pick, PicksResponse

    player_ids = player_ids or list(range(1, 16))
    picks = [
        Pick(
            element=pid,
            position=i + 1,
            multiplier=1,
            is_captain=(i == 0),
            is_vice_captain=(i == 1),
        )
        for i, pid in enumerate(player_ids)
    ]
    return PicksResponse(
        picks=picks,
        entry_history={"bank": 25, "event_transfers": event_transfers, "value": 1000},
    )


def _minimal_reconstructed_state(team_id: int = 44, target_gw: int = 15) -> dict:
    """Return the minimal valid state dict that backtest harness injects."""
    squad_player = {
        "id": 1,
        "web_name": "TestPlayer",
        "element_type": 3,
        "team": 1,
        "team_name": "ARS",
        "now_cost": 80,
        "selling_price": 80,
        "form": "5.0",
        "form_float": 5.0,
        "ep_next": "5.0",
        "ep_next_float": 5.0,
        "total_points": 100,
        "expected_goals": "0.00",
        "expected_assists": "0.00",
        "minutes": 2000,
        "news": "",
        "status": "a",
        "chance_of_playing_next_round": 100,
        "position_name": "MID",
        "is_captain": True,
        "is_vice_captain": False,
        "multiplier": 2,
        "fixture": {"opponent": "MCI", "is_home": True, "difficulty": 4},
    }
    return {
        "team_id": team_id,
        "target_gw": target_gw,
        "current_squad": [squad_player] * 15,
        "bank": 25,
        "free_transfers": 1,
        "candidates": [],
        "fixtures": [],
        "teams": {1: "ARS"},
        "player_form": {1: {"recent_avg_points": 5.0, "recent_minutes": 450, "recent_games": 5}},
        "replan_count": 0,
        "proposed_transfer": None,
        "is_valid": False,
        "violations": [],
        "provider": "openai",
        "force_replan": False,
    }


def _make_final_state(squad_player: dict | None = None) -> dict:
    """Return the minimal final state the graph returns after ainvoke."""
    sp = squad_player or {
        "id": 1,
        "web_name": "TestPlayer",
        "element_type": 3,
        "team": 1,
        "team_name": "ARS",
        "now_cost": 80,
        "selling_price": 80,
        "form": "5.0",
        "form_float": 5.0,
        "ep_next": "5.0",
        "ep_next_float": 5.0,
        "total_points": 100,
        "expected_goals": "0.00",
        "expected_assists": "0.00",
        "minutes": 2000,
        "news": "",
        "status": "a",
        "chance_of_playing_next_round": 100,
        "position_name": "MID",
        "is_captain": False,
        "is_vice_captain": False,
        "multiplier": 1,
        "fixture": {"opponent": "MCI", "is_home": True, "difficulty": 4},
    }
    return {
        "current_squad": [sp] * 15,
        "proposed_transfer": None,
        "captain_pick": {"id": 1, "name": "TestPlayer"},
        "vice_captain_pick": {"id": 2, "name": "OtherPlayer"},
        "transfer_reasoning": "Hold — squad is performing well.",
        "captain_reasoning": "TestPlayer has the best fixture.",
        "recommendation": "Hold. Captain TestPlayer.",
        "replan_count": 0,
        "is_valid": True,
        "bank": 25,
        "free_transfers": 1,
        "candidates": [],
        "fixtures": [],
        "teams": {1: "ARS"},
        "player_form": {},
        "violations": [],
        "provider": "openai",
        "force_replan": False,
    }


# ---------------------------------------------------------------------------
# build_backtest_graph
# ---------------------------------------------------------------------------

class TestBuildBacktestGraph:
    def test_compiles_without_error(self):
        graph = build_backtest_graph()
        assert graph is not None

    def test_has_no_fetch_context_node(self):
        """The backtest graph must start at analyze_and_propose, not fetch_context."""
        graph = build_backtest_graph()
        # The compiled graph's nodes are accessible via .nodes
        node_names = set(graph.nodes.keys())
        assert "fetch_context" not in node_names, (
            "backtest graph must NOT include fetch_context — it injects state directly"
        )
        assert "analyze_and_propose" in node_names


# ---------------------------------------------------------------------------
# run_single_gw
# ---------------------------------------------------------------------------

class TestRunSingleGw:
    async def test_returns_gw_result_on_success(self):
        """run_single_gw returns a GWResult when all components succeed."""
        picks_resp = _make_picks_resp()
        final_state = _make_final_state()
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value=final_state)

        all_histories = {i: [] for i in range(1, 16)}

        with patch("fpl_strategist.eval.backtest.FPLClient") as mock_client_cls, \
             patch("fpl_strategist.eval.backtest.reconstruct_historical_state") as mock_recon, \
             patch("fpl_strategist.eval.backtest.heuristic_recommend") as mock_heuristic:

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get_picks = AsyncMock(return_value=picks_resp)
            mock_client.get_fixtures = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            mock_recon.return_value = _minimal_reconstructed_state()

            mock_heuristic_result = MagicMock()
            mock_heuristic_result.player_out = None
            mock_heuristic_result.player_in = None
            mock_heuristic_result.captain_id = 1
            mock_heuristic_result.captain_name = "TestPlayer"
            mock_heuristic.return_value = mock_heuristic_result

            result = await run_single_gw(
                team_id=44,
                target_gw=15,
                bootstrap_data={"elements": [], "teams": []},
                all_histories=all_histories,
                backtest_graph=mock_graph,
                provider="openai",
                run_judge=False,
                api_semaphore=asyncio.Semaphore(5),
            )

        assert isinstance(result, GWResult)
        assert result.gw == 15

    async def test_returns_none_on_404(self):
        """run_single_gw returns None when picks returns 404 (team has no history for GW)."""
        mock_graph = AsyncMock()
        all_histories = {}

        with patch("fpl_strategist.eval.backtest.FPLClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get_picks = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Not Found",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                )
            )
            mock_client_cls.return_value = mock_client

            result = await run_single_gw(
                team_id=44,
                target_gw=5,
                bootstrap_data={"elements": [], "teams": []},
                all_histories=all_histories,
                backtest_graph=mock_graph,
                provider="openai",
                run_judge=False,
                api_semaphore=asyncio.Semaphore(5),
            )

        assert result is None

    async def test_data_contract_all_fields_populated(self):
        """All GWResult fields have valid values — no None where int expected."""
        picks_resp = _make_picks_resp()
        final_state = _make_final_state()
        final_state["proposed_transfer"] = {
            "out": {"id": 3, "web_name": "OutPlayer", "now_cost": 70, "form_float": 2.0},
            "in":  {"id": 99, "web_name": "InPlayer",  "now_cost": 75, "form_float": 6.0},
        }

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value=final_state)

        all_histories = {
            **{i: [] for i in range(1, 16)},
            3:  [{"round": 15, "total_points": 4, "minutes": 90, "value": 70}],
            99: [{"round": 15, "total_points": 9, "minutes": 90, "value": 75}],
        }

        with patch("fpl_strategist.eval.backtest.FPLClient") as mock_client_cls, \
             patch("fpl_strategist.eval.backtest.reconstruct_historical_state") as mock_recon, \
             patch("fpl_strategist.eval.backtest.heuristic_recommend") as mock_heuristic:

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get_picks = AsyncMock(return_value=picks_resp)
            mock_client.get_fixtures = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            mock_recon.return_value = _minimal_reconstructed_state()

            mock_heuristic_result = MagicMock()
            mock_heuristic_result.player_out = {"id": 3, "web_name": "OutPlayer"}
            mock_heuristic_result.player_in  = {"id": 99, "web_name": "InPlayer"}
            mock_heuristic_result.captain_id = 1
            mock_heuristic_result.captain_name = "TestPlayer"
            mock_heuristic.return_value = mock_heuristic_result

            result = await run_single_gw(
                team_id=44,
                target_gw=15,
                bootstrap_data={"elements": [], "teams": []},
                all_histories=all_histories,
                backtest_graph=mock_graph,
                provider="openai",
                run_judge=False,
                api_semaphore=asyncio.Semaphore(5),
            )

        assert isinstance(result, GWResult)
        # Integer fields must not be None
        assert isinstance(result.gw, int)
        assert isinstance(result.transfer_delta, int)
        assert isinstance(result.captain_delta, int)
        assert isinstance(result.heuristic_transfer_delta, int)
        assert isinstance(result.heuristic_captain_delta, int)
        assert isinstance(result.replan_count, int)
        # Name fields must be non-empty strings
        assert result.player_out_name != ""
        assert result.player_in_name != ""
        assert result.agent_captain_name != ""
        # Transfer action must be one of the expected values
        assert result.transfer_action in ("transfer", "hold")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes so CI Rich output is plain-text comparable."""
    import re
    return re.sub(r"\x1b\[[0-9;]*[mK]", "", text)


class TestBacktestCli:
    def test_backtest_command_exists(self):
        """Verify the CLI has a 'backtest' command with the required options."""
        from typer.testing import CliRunner
        from fpl_strategist.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["backtest", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "--from-gw" in plain
        assert "--to-gw" in plain
        assert "--skip-judge" in plain
        assert "--json-only" in plain
