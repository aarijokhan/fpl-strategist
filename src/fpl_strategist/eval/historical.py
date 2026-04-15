"""Historical state reconstruction for backtesting.

Builds a fetch_context-compatible state dict from cached player histories
and API data, preventing data leakage by filtering to round < target_gw.

Limitations (acknowledged):
- ep_next approximated as computed 5-GW form average (proprietary to FPL)
- status/news not historically reconstructable; all players set to available
- Availability constraint (Rule 7) will not fire during backtests
- Selling price remains now_cost approximation (same as live MVP)
"""

from __future__ import annotations

import asyncio
import logging
import warnings

from fpl_strategist.data.candidate_filter import filter_candidates
from fpl_strategist.data.fpl_client import FPLClient
from fpl_strategist.data.models import Fixture, Player

logger = logging.getLogger(__name__)


async def prefetch_all_histories(
    client: FPLClient,
    player_ids: list[int],
    semaphore: asyncio.Semaphore,
) -> dict[int, list[dict]]:
    """Fetch per-GW history for all players with concurrency control.

    Returns {player_id: [history_entry, ...]} where each entry has at least
    'round', 'total_points', 'minutes', 'value' (price at that GW).
    Players whose fetch fails are skipped with a warning.
    """
    async def _fetch_one(pid: int) -> tuple[int, list[dict]]:
        async with semaphore:
            try:
                summary = await client.get_player_summary(pid)
                return pid, summary.get("history", [])
            except Exception as exc:
                logger.warning("Failed to fetch summary for player %d: %s", pid, exc)
                return pid, []

    tasks = [_fetch_one(pid) for pid in player_ids]
    pairs = await asyncio.gather(*tasks)
    return {pid: history for pid, history in pairs}


def _compute_form(history: list[dict], target_gw: int) -> float:
    """Compute average total_points over the 5 GWs immediately before target_gw.

    Only includes rounds in [target_gw-5, target_gw-1]. Returns 0.0 if no data.
    Strictly excludes round >= target_gw to prevent data leakage.
    """
    window = [
        h for h in history
        if target_gw - 5 <= h.get("round", 0) < target_gw
    ]
    if not window:
        return 0.0
    return sum(h.get("total_points", 0) for h in window) / len(window)


def _historical_price(history: list[dict], target_gw: int, fallback: int) -> int:
    """Return the player's price (in tenths) at target_gw - 1.

    Uses the 'value' field from the history entry at round == target_gw - 1.
    Falls back to the current bootstrap price if no entry found.
    """
    for entry in history:
        if entry.get("round") == target_gw - 1:
            return entry.get("value", fallback)
    return fallback


def reconstruct_historical_state(
    team_id: int,
    target_gw: int,
    bootstrap_data: dict,
    all_histories: dict[int, list[dict]],
    picks_resp,
    fixtures: list,
) -> dict:
    """Build a fetch_context-compatible state dict for a historical gameweek.

    Args:
        team_id: FPL team ID.
        target_gw: The gameweek being backtested (the one the agent recommends for).
        bootstrap_data: Raw response from /bootstrap-static/ (for player IDs, team names).
        all_histories: {player_id: [history_entries]} prefetched for all players.
            Each entry must have 'round', 'total_points', 'minutes', 'value'.
        picks_resp: PicksResponse from get_picks(team_id, target_gw - 1).
            This gives the squad and bank BEFORE the target gameweek.
            Its entry_history["event_transfers"] records transfers made in GW N-1,
            which determines free transfers available for GW N.
        fixtures: list of Fixture objects for target_gw.

    Returns:
        A dict matching the exact shape returned by fetch_context, suitable
        for injection directly into analyze_and_propose.
    """
    teams_lookup: dict[int, str] = {
        t["id"]: t["short_name"] for t in bootstrap_data.get("teams", [])
    }
    players_by_id: dict[int, dict] = {
        p["id"]: p for p in bootstrap_data.get("elements", [])
    }

    # Bank and free transfers from the prior-gameweek picks
    # IMPORTANT: entry_history here is from GW N-1 (target_gw - 1).
    # event_transfers = number of transfers the manager made in GW N-1,
    # which determines how many free transfers they have for GW N.
    bank = picks_resp.entry_history.get("bank", 0)
    event_transfers = picks_resp.entry_history.get("event_transfers", 0)
    free_transfers = min(2, max(1, 2 - event_transfers)) if target_gw > 1 else 1

    # Build fixture display lookup: team_id -> {opponent, is_home, difficulty}
    fixture_display: dict[int, dict] = {}
    for fix in fixtures:
        fixture_display[fix.team_h] = {
            "opponent": teams_lookup.get(fix.team_a, "???"),
            "is_home": True,
            "difficulty": fix.team_h_difficulty,
        }
        fixture_display[fix.team_a] = {
            "opponent": teams_lookup.get(fix.team_h, "???"),
            "is_home": False,
            "difficulty": fix.team_a_difficulty,
        }

    # Enrich squad players with historical data
    pos_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    current_squad: list[dict] = []
    player_form: dict[int, dict] = {}

    for pick in picks_resp.picks:
        p_data = players_by_id.get(pick.element)
        if p_data is None:
            continue

        history = all_histories.get(pick.element, [])
        form_val = _compute_form(history, target_gw)
        price = _historical_price(history, target_gw, fallback=p_data.get("now_cost", 0))
        fix_info = fixture_display.get(p_data.get("team", 0), {})

        form_str = f"{form_val:.1f}"

        squad_entry = {
            "id": pick.element,
            "web_name": p_data.get("web_name", ""),
            "element_type": p_data.get("element_type", 0),
            "team": p_data.get("team", 0),
            "team_name": teams_lookup.get(p_data.get("team", 0), "???"),
            "now_cost": price,
            "selling_price": price,
            # Historical form (no leakage)
            "form": form_str,
            "form_float": round(form_val, 1),
            # ep_next approximated as computed form
            "ep_next": form_str,
            "ep_next_float": round(form_val, 1),
            # Season totals from bootstrap (current snapshot — acceptable for backtesting)
            "total_points": p_data.get("total_points", 0),
            "expected_goals": p_data.get("expected_goals", "0.00"),
            "expected_assists": p_data.get("expected_assists", "0.00"),
            "minutes": p_data.get("minutes", 0),
            # Availability: not reconstructable historically — set all available
            "status": "a",
            "news": "",
            "chance_of_playing_next_round": 100,
            "position_name": pos_names.get(p_data.get("element_type", 0), "UNK"),
            "is_captain": pick.is_captain,
            "is_vice_captain": pick.is_vice_captain,
            "multiplier": pick.multiplier,
            "fixture": fix_info,
        }
        current_squad.append(squad_entry)

        # Also build player_form for this player
        recent = [
            h for h in history
            if target_gw - 5 <= h.get("round", 0) < target_gw
        ]
        total_mins = sum(h.get("minutes", 0) for h in recent)
        player_form[pick.element] = {
            "recent_avg_points": round(form_val, 1),
            "recent_minutes": total_mins,
            "recent_games": len(recent),
        }

    # Build candidate list: reconstruct all players with historical data,
    # run candidate_filter (same as fetch_context live path)
    squad_ids = {pick.element for pick in picks_resp.picks}

    reconstructed_players: list[Player] = []
    for p_data in bootstrap_data.get("elements", []):
        pid = p_data["id"]
        history = all_histories.get(pid, [])
        form_val = _compute_form(history, target_gw)
        price = _historical_price(history, target_gw, fallback=p_data.get("now_cost", 0))

        # Build a patched player dict with historical values
        patched = dict(p_data)
        patched["now_cost"] = price
        patched["form"] = f"{form_val:.1f}"
        patched["ep_next"] = f"{form_val:.1f}"
        patched["status"] = "a"
        patched["news"] = ""
        patched["chance_of_playing_next_round"] = 100

        try:
            reconstructed_players.append(Player(**patched))
        except Exception:
            # Skip players with missing required fields
            continue

    fixture_models = fixtures  # already Fixture objects
    candidates_models = filter_candidates(
        all_players=reconstructed_players,
        squad_ids=squad_ids,
        fixtures=fixture_models,
        teams=teams_lookup,
    )

    candidates: list[dict] = []
    for c in candidates_models:
        fix_info = fixture_display.get(c.team, {})
        candidates.append({
            "id": c.id,
            "web_name": c.web_name,
            "element_type": c.element_type,
            "team": c.team,
            "team_name": teams_lookup.get(c.team, "???"),
            "now_cost": c.now_cost,
            "form": c.form,
            "form_float": c.form_float,
            "ep_next": c.ep_next,
            "ep_next_float": c.ep_next_float,
            "total_points": c.total_points,
            "status": c.status,
            "news": c.news,
            "position_name": c.position_name,
            "fixture": fix_info,
        })

    return {
        "current_squad": current_squad,
        "bank": bank,
        "free_transfers": free_transfers,
        "candidates": candidates,
        "fixtures": [f.model_dump() for f in fixtures],
        "teams": teams_lookup,
        "player_form": player_form,
        "replan_count": 0,
        "proposed_transfer": None,
        "is_valid": False,
        "violations": [],
    }
