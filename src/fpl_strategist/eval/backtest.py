"""Backtest harness for Phase 6 evaluation.

Replays historical gameweeks: reconstructs pre-GW state, runs the agent
and heuristic baseline, scores outcomes against actual results.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
from langgraph.graph import END, START, StateGraph

from fpl_strategist.data.fpl_client import FPLClient
from fpl_strategist.eval.heuristic import heuristic_recommend
from fpl_strategist.eval.historical import prefetch_all_histories, reconstruct_historical_state
from fpl_strategist.eval.judge import judge_coherence
from fpl_strategist.eval.scoring import (
    GWResult,
    BacktestSummary,
    lookup_actual_points,
    results_to_table,
    score_captain,
    score_transfer,
    summarize_results,
)
from fpl_strategist.nodes.analyze import analyze_and_propose
from fpl_strategist.nodes.explain import explain_recommendation
from fpl_strategist.nodes.replan import replan_transfer
from fpl_strategist.nodes.select_captain import select_captain
from fpl_strategist.nodes.validate import validate_constraints
from fpl_strategist.graph import (
    MAX_REPLANS,
    _clear_proposal_if_exhausted,
    _route_after_validation,
)
from fpl_strategist.state import FPLState

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "results"


def _get_results_dir() -> Path:
    """Return the eval/results directory, creating it if needed."""
    # Walk up from this file to the project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    results_dir = project_root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


async def select_captain_with_fallback(state: FPLState) -> dict:
    """Wrapper that clears proposal if replans exhausted, then selects captain."""
    cleared = _clear_proposal_if_exhausted(state)
    captain_result = await select_captain(state)
    return {**cleared, **captain_result}


def build_backtest_graph():
    """Build a graph that starts at analyze_and_propose (no fetch_context).

    The backtest harness provides pre-populated state directly, so
    fetch_context is omitted. All other nodes and routing are identical
    to the live graph.
    """
    builder = StateGraph(FPLState)

    builder.add_node("analyze_and_propose", analyze_and_propose)
    builder.add_node("validate_constraints", validate_constraints)
    builder.add_node("replan_transfer", replan_transfer)
    builder.add_node("select_captain", select_captain_with_fallback)
    builder.add_node("explain_recommendation", explain_recommendation)

    builder.add_edge(START, "analyze_and_propose")
    builder.add_edge("analyze_and_propose", "validate_constraints")

    builder.add_conditional_edges(
        "validate_constraints",
        _route_after_validation,
        {
            "select_captain": "select_captain",
            "replan_transfer": "replan_transfer",
        },
    )

    builder.add_edge("replan_transfer", "validate_constraints")
    builder.add_edge("select_captain", "explain_recommendation")
    builder.add_edge("explain_recommendation", END)

    return builder.compile()


async def run_single_gw(
    team_id: int,
    target_gw: int,
    bootstrap_data: dict,
    all_histories: dict[int, list[dict]],
    backtest_graph,
    provider: str,
    run_judge: bool,
    api_semaphore: asyncio.Semaphore,
) -> GWResult | None:
    """Run agent and heuristic for one historical gameweek.

    Returns None if the GW is unavailable (404) or fatally errored.
    """
    try:
        async with api_semaphore:
            async with FPLClient() as client:
                picks_resp = await client.get_picks(team_id, target_gw - 1)
                raw_fixtures = await client.get_fixtures(target_gw)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.info("GW %d: team %d has no picks (404) — skipping.", target_gw, team_id)
            return None
        logger.warning("GW %d: HTTP error %d — skipping.", target_gw, exc.response.status_code)
        return None
    except Exception as exc:
        logger.warning("GW %d: unexpected error fetching picks: %s — skipping.", target_gw, exc)
        return None

    # Reconstruct pre-GW state
    state = reconstruct_historical_state(
        team_id=team_id,
        target_gw=target_gw,
        bootstrap_data=bootstrap_data,
        all_histories=all_histories,
        picks_resp=picks_resp,
        fixtures=raw_fixtures,
    )
    state["team_id"] = team_id
    state["target_gw"] = target_gw
    state["provider"] = provider
    state["force_replan"] = False

    # Heuristic baseline
    heuristic = heuristic_recommend(
        state["current_squad"],
        state["candidates"],
        state["bank"],
        state["fixtures"],
        state["teams"],
        use_fixtures=True,
    )

    # Run agent
    try:
        final_state = await backtest_graph.ainvoke(state)
    except Exception as exc:
        logger.warning("GW %d: agent error: %s — skipping.", target_gw, exc)
        return None

    # Determine actual user captain from prior picks
    user_captain_id: int = 0
    user_captain_name: str = "Unknown"
    players_by_id = {p["id"]: p for p in state["current_squad"]}
    for pick in picks_resp.picks:
        if pick.is_captain:
            p = players_by_id.get(pick.element, {})
            user_captain_id = pick.element
            user_captain_name = p.get("web_name", "Unknown")
            break

    # Score agent transfer
    agent_transfer = final_state.get("proposed_transfer")
    if agent_transfer:
        out_id = agent_transfer["out"]["id"]
        in_id = agent_transfer["in"]["id"]
        out_name = agent_transfer["out"]["web_name"]
        in_name = agent_transfer["in"]["web_name"]
        out_pts = lookup_actual_points(out_id, target_gw, all_histories)
        in_pts = lookup_actual_points(in_id, target_gw, all_histories)
        transfer_delta = score_transfer(out_pts, in_pts)
        transfer_action = "transfer"
    else:
        out_id = in_id = None
        out_name = in_name = "--"
        out_pts = in_pts = 0
        transfer_delta = 0
        transfer_action = "hold"

    # Score agent captain
    agent_cap = final_state.get("captain_pick") or {}
    agent_cap_id = agent_cap.get("id", 0)
    agent_cap_name = agent_cap.get("name", "Unknown")
    agent_cap_pts = lookup_actual_points(agent_cap_id, target_gw, all_histories)
    user_cap_pts = lookup_actual_points(user_captain_id, target_gw, all_histories)
    captain_delta = score_captain(agent_cap_pts, user_cap_pts)

    # Score heuristic transfer
    if heuristic.player_out and heuristic.player_in:
        h_out_pts = lookup_actual_points(heuristic.player_out["id"], target_gw, all_histories)
        h_in_pts = lookup_actual_points(heuristic.player_in["id"], target_gw, all_histories)
        h_transfer_delta = score_transfer(h_out_pts, h_in_pts)
        h_out_name = heuristic.player_out["web_name"]
        h_in_name = heuristic.player_in["web_name"]
    else:
        h_transfer_delta = 0
        h_out_name = h_in_name = "--"

    h_cap_pts = lookup_actual_points(heuristic.captain_id, target_gw, all_histories)
    h_captain_delta = score_captain(h_cap_pts, user_cap_pts)

    # LLM-judge
    coherence_score = factual = coherence = actionability = None
    if run_judge:
        try:
            judge = await judge_coherence(final_state)
            coherence_score = judge.average
            factual = float(judge.factual_grounding)
            coherence = float(judge.logical_coherence)
            actionability = float(judge.actionability)
        except Exception as exc:
            logger.warning("GW %d: judge error: %s", target_gw, exc)

    first_valid = final_state.get("replan_count", 0) == 0 and final_state.get("is_valid", False)

    return GWResult(
        gw=target_gw,
        transfer_action=transfer_action,
        player_out_id=out_id,
        player_in_id=in_id,
        player_out_name=out_name,
        player_in_name=in_name,
        player_out_actual_pts=out_pts,
        player_in_actual_pts=in_pts,
        transfer_delta=transfer_delta,
        agent_captain_id=agent_cap_id,
        agent_captain_name=agent_cap_name,
        agent_captain_actual_pts=agent_cap_pts,
        user_captain_id=user_captain_id,
        user_captain_name=user_captain_name,
        user_captain_actual_pts=user_cap_pts,
        captain_delta=captain_delta,
        replan_count=final_state.get("replan_count", 0),
        is_valid_first_attempt=first_valid,
        heuristic_transfer_delta=h_transfer_delta,
        heuristic_captain_delta=h_captain_delta,
        heuristic_out_name=h_out_name,
        heuristic_in_name=h_in_name,
        heuristic_captain_name=heuristic.captain_name,
        coherence_score=coherence_score,
        factual_grounding=factual,
        logical_coherence=coherence,
        actionability=actionability,
    )


async def run_backtest(
    team_id: int,
    from_gw: int,
    to_gw: int,
    provider: str = "openai",
    skip_judge: bool = False,
    concurrency: int = 5,
) -> tuple[list[GWResult], BacktestSummary]:
    """Run full backtest across a GW range.

    Prefetches all player histories once, then iterates GWs sequentially
    (LLM calls are the bottleneck; no benefit to parallelising GWs).
    """
    api_semaphore = asyncio.Semaphore(concurrency)

    async with FPLClient() as client:
        bootstrap_data = await client.get_bootstrap()
        player_ids = [p["id"] for p in bootstrap_data.get("elements", [])]
        all_histories = await prefetch_all_histories(client, player_ids, api_semaphore)

    backtest_graph = build_backtest_graph()

    results: list[GWResult] = []
    for gw in range(from_gw, to_gw + 1):
        result = await run_single_gw(
            team_id=team_id,
            target_gw=gw,
            bootstrap_data=bootstrap_data,
            all_histories=all_histories,
            backtest_graph=backtest_graph,
            provider=provider,
            run_judge=not skip_judge,
            api_semaphore=api_semaphore,
        )
        if result is not None:
            results.append(result)

    summary = summarize_results(results, team_id=team_id, gw_range=(from_gw, to_gw))
    return results, summary


def save_results_json(
    results: list[GWResult],
    summary: BacktestSummary,
    team_id: int,
    from_gw: int,
    to_gw: int,
) -> Path:
    """Save backtest results to eval/results/ as a JSON file."""
    results_dir = _get_results_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"backtest_{team_id}_gw{from_gw}-{to_gw}_{ts}.json"
    path = results_dir / filename

    data = {
        "summary": {
            "team_id": summary.team_id,
            "gw_range": list(summary.gw_range),
            "total_gws": summary.total_gws,
            "agent_avg_transfer_delta": summary.agent_avg_transfer_delta,
            "agent_transfer_delta_std": summary.agent_transfer_delta_std,
            "agent_transfer_hit_rate": summary.agent_transfer_hit_rate,
            "heuristic_avg_transfer_delta": summary.heuristic_avg_transfer_delta,
            "heuristic_transfer_delta_std": summary.heuristic_transfer_delta_std,
            "heuristic_transfer_hit_rate": summary.heuristic_transfer_hit_rate,
            "agent_avg_captain_delta": summary.agent_avg_captain_delta,
            "agent_captain_delta_std": summary.agent_captain_delta_std,
            "agent_captain_hit_rate": summary.agent_captain_hit_rate,
            "heuristic_avg_captain_delta": summary.heuristic_avg_captain_delta,
            "heuristic_captain_delta_std": summary.heuristic_captain_delta_std,
            "heuristic_captain_hit_rate": summary.heuristic_captain_hit_rate,
            "avg_coherence": summary.avg_coherence,
            "coherence_std": summary.coherence_std,
            "avg_replan_count": summary.avg_replan_count,
            "first_attempt_valid_rate": summary.first_attempt_valid_rate,
            "hold_rate": summary.hold_rate,
        },
        "results": results_to_table(results),
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return path
