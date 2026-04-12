"""fetch_context node — deterministic, no LLM.

Calls the FPL API, enriches the squad with form/fixture data,
runs the candidate filter, and populates the data layer in state.
"""

from __future__ import annotations

from fpl_strategist.data.candidate_filter import filter_candidates
from fpl_strategist.data.fpl_client import FPLClient
from fpl_strategist.data.models import Fixture, Player
from fpl_strategist.state import FPLState


async def fetch_context(state: FPLState) -> dict:
    """Fetch all FPL data needed for transfer analysis."""
    team_id = state["team_id"]
    target_gw = state["target_gw"]

    async with FPLClient() as client:
        # Parallel-safe: bootstrap is a single large request, others are small
        bootstrap = await client.get_bootstrap()

        # Build lookups from bootstrap
        teams_lookup: dict[int, str] = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
        players_by_id: dict[int, dict] = {p["id"]: p for p in bootstrap["elements"]}

        # Fetch squad picks from the most recent completed gameweek
        prev_gw = target_gw - 1 if target_gw > 1 else target_gw
        picks_resp = await client.get_picks(team_id, prev_gw)

        bank = picks_resp.entry_history.get("bank", 0)
        event_transfers = picks_resp.entry_history.get("event_transfers", 0)
        free_transfers = min(2, max(1, 2 - event_transfers)) if target_gw > 1 else 1

        # Fetch fixtures for target gameweek
        raw_fixtures = await client.get_fixtures(target_gw)
        fixtures = [f.model_dump() for f in raw_fixtures]

        # Build fixture lookup: team_id -> fixture difficulty info
        fixture_display: dict[int, dict] = {}
        for fix in raw_fixtures:
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

        # Enrich squad players
        squad_player_ids = [pick.element for pick in picks_resp.picks]
        player_summaries = await client.get_player_summaries(squad_player_ids)

        current_squad: list[dict] = []
        for pick in picks_resp.picks:
            p_data = players_by_id.get(pick.element)
            if p_data is None:
                continue

            player = Player(**p_data)
            fix_info = fixture_display.get(player.team, {})

            squad_entry = {
                "id": player.id,
                "web_name": player.web_name,
                "element_type": player.element_type,
                "team": player.team,
                "team_name": teams_lookup.get(player.team, "???"),
                "now_cost": player.now_cost,
                "selling_price": player.now_cost,  # MVP approximation
                "form": player.form,
                "form_float": player.form_float,
                "total_points": player.total_points,
                "ep_next": player.ep_next,
                "ep_next_float": player.ep_next_float,
                "expected_goals": player.expected_goals,
                "expected_assists": player.expected_assists,
                "minutes": player.minutes,
                "news": player.news,
                "status": player.status,
                "chance_of_playing_next_round": player.chance_of_playing_next_round,
                "position_name": player.position_name,
                "is_captain": pick.is_captain,
                "is_vice_captain": pick.is_vice_captain,
                "multiplier": pick.multiplier,
                "fixture": fix_info,
            }
            current_squad.append(squad_entry)

        # Build player form from summaries
        player_form: dict[int, dict] = {}
        for pid, summary in player_summaries.items():
            history = summary.get("history", [])
            recent = [h for h in history if h.get("round", 0) >= target_gw - 5]
            if recent:
                avg_pts = sum(h.get("total_points", 0) for h in recent) / len(recent)
                total_mins = sum(h.get("minutes", 0) for h in recent)
            else:
                avg_pts = 0.0
                total_mins = 0
            player_form[pid] = {
                "recent_avg_points": round(avg_pts, 1),
                "recent_minutes": total_mins,
                "recent_games": len(recent),
            }

        # Run candidate filter
        all_players = [Player(**p) for p in bootstrap["elements"]]
        squad_ids = {pick.element for pick in picks_resp.picks}
        fixture_models = [Fixture(**f) for f in fixtures]

        candidates_models = filter_candidates(
            all_players=all_players,
            squad_ids=squad_ids,
            fixtures=fixture_models,
            teams=teams_lookup,
        )

        # Convert candidates to dicts with fixture info
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
        "fixtures": fixtures,
        "teams": teams_lookup,
        "player_form": player_form,
        "replan_count": 0,
        "proposed_transfer": None,
        "is_valid": False,
        "violations": [],
    }
