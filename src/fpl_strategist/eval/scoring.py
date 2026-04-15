"""Scoring module for Phase 6 evaluation.

Data classes and pure functions for measuring agent performance
against heuristic baselines on historical gameweeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GWResult:
    """One row of backtest output — results for a single historical gameweek."""

    # Gameweek
    gw: int

    # Transfer
    transfer_action: str  # "transfer" or "hold"
    player_out_id: int | None
    player_in_id: int | None
    player_out_name: str
    player_in_name: str
    player_out_actual_pts: int
    player_in_actual_pts: int
    transfer_delta: int  # player_in_actual_pts - player_out_actual_pts (0 for hold)

    # Captain
    agent_captain_id: int
    agent_captain_name: str
    agent_captain_actual_pts: int  # raw points before doubling
    user_captain_id: int
    user_captain_name: str
    user_captain_actual_pts: int  # raw points before doubling
    captain_delta: int  # agent_captain - user_captain (raw)

    # Validation
    replan_count: int
    is_valid_first_attempt: bool  # True if first proposal passed constraints

    # Heuristic baseline (form × fixture)
    heuristic_transfer_delta: int
    heuristic_captain_delta: int
    heuristic_out_name: str
    heuristic_in_name: str
    heuristic_captain_name: str

    # LLM-judge scores (None if --skip-judge)
    coherence_score: float | None = None
    factual_grounding: float | None = None
    logical_coherence: float | None = None
    actionability: float | None = None


@dataclass
class BacktestSummary:
    """Aggregate statistics across all backtested gameweeks."""

    team_id: int
    gw_range: tuple[int, int]
    total_gws: int

    # Agent transfer metrics
    agent_avg_transfer_delta: float
    agent_transfer_delta_std: float
    agent_transfer_hit_rate: float  # % GWs where transfer_delta > 0

    # Heuristic transfer metrics
    heuristic_avg_transfer_delta: float
    heuristic_transfer_delta_std: float
    heuristic_transfer_hit_rate: float

    # Agent captain metrics
    agent_avg_captain_delta: float
    agent_captain_delta_std: float
    agent_captain_hit_rate: float  # % GWs where captain_delta > 0

    # Heuristic captain metrics
    heuristic_avg_captain_delta: float
    heuristic_captain_delta_std: float
    heuristic_captain_hit_rate: float

    # Coherence (None if judge was skipped)
    avg_coherence: float | None
    coherence_std: float | None

    # Process metrics
    avg_replan_count: float
    first_attempt_valid_rate: float  # % GWs where first proposal was valid
    hold_rate: float  # % GWs where agent recommended hold


def score_transfer(player_out_pts: int, player_in_pts: int) -> int:
    """Return player_in_pts - player_out_pts. Use 0 for both when holding."""
    return player_in_pts - player_out_pts


def score_captain(agent_captain_pts: int, user_captain_pts: int) -> int:
    """Return agent_captain_pts - user_captain_pts (raw, before doubling)."""
    return agent_captain_pts - user_captain_pts


def lookup_actual_points(
    player_id: int,
    gw: int,
    all_histories: dict[int, list[dict]],
) -> int:
    """Return a player's actual points in a specific gameweek.

    Searches the player's history list for an entry with round == gw.
    Returns 0 if the player is not found or didn't play that GW.
    """
    history = all_histories.get(player_id, [])
    for entry in history:
        if entry.get("round") == gw:
            return entry.get("total_points", 0)
    return 0


def _std(values: list[float]) -> float:
    """Population standard deviation of a list. Returns 0.0 for empty/single-element."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def summarize_results(
    results: list[GWResult],
    team_id: int,
    gw_range: tuple[int, int],
) -> BacktestSummary:
    """Compute aggregate statistics from a list of GWResult records."""
    n = len(results)
    if n == 0:
        return BacktestSummary(
            team_id=team_id,
            gw_range=gw_range,
            total_gws=0,
            agent_avg_transfer_delta=0.0,
            agent_transfer_delta_std=0.0,
            agent_transfer_hit_rate=0.0,
            heuristic_avg_transfer_delta=0.0,
            heuristic_transfer_delta_std=0.0,
            heuristic_transfer_hit_rate=0.0,
            agent_avg_captain_delta=0.0,
            agent_captain_delta_std=0.0,
            agent_captain_hit_rate=0.0,
            heuristic_avg_captain_delta=0.0,
            heuristic_captain_delta_std=0.0,
            heuristic_captain_hit_rate=0.0,
            avg_coherence=None,
            coherence_std=None,
            avg_replan_count=0.0,
            first_attempt_valid_rate=0.0,
            hold_rate=0.0,
        )

    agent_td = [float(r.transfer_delta) for r in results]
    heuristic_td = [float(r.heuristic_transfer_delta) for r in results]
    agent_cd = [float(r.captain_delta) for r in results]
    heuristic_cd = [float(r.heuristic_captain_delta) for r in results]

    coherence_vals = [r.coherence_score for r in results if r.coherence_score is not None]

    return BacktestSummary(
        team_id=team_id,
        gw_range=gw_range,
        total_gws=n,
        agent_avg_transfer_delta=sum(agent_td) / n,
        agent_transfer_delta_std=_std(agent_td),
        agent_transfer_hit_rate=sum(1 for v in agent_td if v > 0) / n,
        heuristic_avg_transfer_delta=sum(heuristic_td) / n,
        heuristic_transfer_delta_std=_std(heuristic_td),
        heuristic_transfer_hit_rate=sum(1 for v in heuristic_td if v > 0) / n,
        agent_avg_captain_delta=sum(agent_cd) / n,
        agent_captain_delta_std=_std(agent_cd),
        agent_captain_hit_rate=sum(1 for v in agent_cd if v > 0) / n,
        heuristic_avg_captain_delta=sum(heuristic_cd) / n,
        heuristic_captain_delta_std=_std(heuristic_cd),
        heuristic_captain_hit_rate=sum(1 for v in heuristic_cd if v > 0) / n,
        avg_coherence=sum(coherence_vals) / len(coherence_vals) if coherence_vals else None,
        coherence_std=_std(coherence_vals) if coherence_vals else None,
        avg_replan_count=sum(r.replan_count for r in results) / n,
        first_attempt_valid_rate=sum(1 for r in results if r.is_valid_first_attempt) / n,
        hold_rate=sum(1 for r in results if r.transfer_action == "hold") / n,
    )


def results_to_table(results: list[GWResult]) -> list[dict]:
    """Flatten each GWResult to a dict for Rich table rendering or JSON export."""
    rows = []
    for r in results:
        rows.append({
            "gw": r.gw,
            "agent_action": r.transfer_action,
            "agent_out": r.player_out_name,
            "agent_in": r.player_in_name,
            "transfer_delta": r.transfer_delta,
            "agent_captain": r.agent_captain_name,
            "agent_cap_pts": r.agent_captain_actual_pts,
            "user_captain": r.user_captain_name,
            "user_cap_pts": r.user_captain_actual_pts,
            "cap_delta": r.captain_delta,
            "h_out": r.heuristic_out_name,
            "h_in": r.heuristic_in_name,
            "h_transfer_delta": r.heuristic_transfer_delta,
            "h_captain": r.heuristic_captain_name,
            "h_cap_delta": r.heuristic_captain_delta,
            "coherence": r.coherence_score,
            "replans": r.replan_count,
            "valid_first": r.is_valid_first_attempt,
        })
    return rows
