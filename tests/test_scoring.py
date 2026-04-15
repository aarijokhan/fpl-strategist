"""Tests for eval/scoring.py — pure scoring functions and data classes."""

from __future__ import annotations

import math

import pytest

from fpl_strategist.eval.scoring import (
    BacktestSummary,
    GWResult,
    lookup_actual_points,
    results_to_table,
    score_captain,
    score_transfer,
    summarize_results,
)


def _make_result(
    gw: int = 10,
    transfer_action: str = "transfer",
    transfer_delta: int = 5,
    captain_delta: int = 3,
    heuristic_transfer_delta: int = 2,
    heuristic_captain_delta: int = 1,
    replan_count: int = 0,
    is_valid_first_attempt: bool = True,
    coherence_score: float | None = None,
) -> GWResult:
    return GWResult(
        gw=gw,
        transfer_action=transfer_action,
        player_out_id=1,
        player_in_id=2,
        player_out_name="Elanga",
        player_in_name="Diaz",
        player_out_actual_pts=2,
        player_in_actual_pts=2 + transfer_delta,
        transfer_delta=transfer_delta,
        agent_captain_id=3,
        agent_captain_name="Salah",
        agent_captain_actual_pts=10 + captain_delta,
        user_captain_id=4,
        user_captain_name="Haaland",
        user_captain_actual_pts=10,
        captain_delta=captain_delta,
        replan_count=replan_count,
        is_valid_first_attempt=is_valid_first_attempt,
        heuristic_transfer_delta=heuristic_transfer_delta,
        heuristic_captain_delta=heuristic_captain_delta,
        heuristic_out_name="Mitchell",
        heuristic_in_name="Gordon",
        heuristic_captain_name="Salah",
        coherence_score=coherence_score,
        factual_grounding=None,
        logical_coherence=None,
        actionability=None,
    )


class TestScoreTransfer:
    def test_positive_delta(self):
        assert score_transfer(2, 9) == 7

    def test_negative_delta(self):
        assert score_transfer(8, 3) == -5

    def test_hold_is_zero(self):
        assert score_transfer(0, 0) == 0

    def test_equal_scores(self):
        assert score_transfer(5, 5) == 0


class TestScoreCaptain:
    def test_positive_delta(self):
        assert score_captain(12, 8) == 4

    def test_negative_delta(self):
        assert score_captain(2, 10) == -8


class TestLookupActualPoints:
    def test_found(self):
        histories = {1: [{"round": 10, "total_points": 8}, {"round": 11, "total_points": 2}]}
        assert lookup_actual_points(1, 10, histories) == 8

    def test_different_gw(self):
        histories = {1: [{"round": 10, "total_points": 8}, {"round": 11, "total_points": 2}]}
        assert lookup_actual_points(1, 11, histories) == 2

    def test_missing_player_returns_zero(self):
        assert lookup_actual_points(999, 10, {}) == 0

    def test_missing_gw_returns_zero(self):
        histories = {1: [{"round": 10, "total_points": 8}]}
        assert lookup_actual_points(1, 99, histories) == 0


class TestSummarizeResults:
    def test_hit_rates(self):
        results = [
            _make_result(gw=10, transfer_delta=5, heuristic_transfer_delta=3),
            _make_result(gw=11, transfer_delta=-2, heuristic_transfer_delta=1),
            _make_result(gw=12, transfer_delta=3, heuristic_transfer_delta=-1),
        ]
        s = summarize_results(results, team_id=44, gw_range=(10, 12))
        assert s.total_gws == 3
        assert abs(s.agent_transfer_hit_rate - 2 / 3) < 0.01
        assert abs(s.heuristic_transfer_hit_rate - 2 / 3) < 0.01

    def test_averages(self):
        results = [
            _make_result(transfer_delta=6, captain_delta=4),
            _make_result(transfer_delta=0, captain_delta=2),
        ]
        s = summarize_results(results, team_id=44, gw_range=(10, 11))
        assert abs(s.agent_avg_transfer_delta - 3.0) < 0.01
        assert abs(s.agent_avg_captain_delta - 3.0) < 0.01

    def test_std_devs(self):
        results = [
            _make_result(transfer_delta=2),
            _make_result(transfer_delta=4),
            _make_result(transfer_delta=6),
        ]
        s = summarize_results(results, team_id=44, gw_range=(10, 12))
        # mean=4, deviations=[-2,0,2], variance=8/3, std≈1.633
        assert abs(s.agent_transfer_delta_std - math.sqrt(8 / 3)) < 0.01

    def test_coherence_aggregated(self):
        results = [
            _make_result(coherence_score=4.0),
            _make_result(coherence_score=3.0),
        ]
        s = summarize_results(results, team_id=44, gw_range=(10, 11))
        assert abs(s.avg_coherence - 3.5) < 0.01

    def test_coherence_none_when_skipped(self):
        results = [_make_result(coherence_score=None)]
        s = summarize_results(results, team_id=44, gw_range=(10, 10))
        assert s.avg_coherence is None

    def test_hold_rate(self):
        results = [
            _make_result(transfer_action="hold"),
            _make_result(transfer_action="transfer"),
            _make_result(transfer_action="transfer"),
        ]
        s = summarize_results(results, team_id=44, gw_range=(10, 12))
        assert abs(s.hold_rate - 1 / 3) < 0.01

    def test_empty_results(self):
        s = summarize_results([], team_id=44, gw_range=(10, 25))
        assert s.total_gws == 0
        assert s.agent_avg_transfer_delta == 0.0


class TestResultsToTable:
    def test_columns_present(self):
        result = _make_result()
        rows = results_to_table([result])
        assert len(rows) == 1
        row = rows[0]
        expected_keys = {
            "gw", "agent_action", "agent_out", "agent_in", "transfer_delta",
            "agent_captain", "agent_cap_pts", "user_captain", "user_cap_pts",
            "cap_delta", "h_out", "h_in", "h_transfer_delta", "h_captain",
            "h_cap_delta", "coherence", "replans", "valid_first",
        }
        assert expected_keys <= set(row.keys())

    def test_values_match_result(self):
        result = _make_result(gw=15, transfer_delta=7)
        rows = results_to_table([result])
        assert rows[0]["gw"] == 15
        assert rows[0]["transfer_delta"] == 7
