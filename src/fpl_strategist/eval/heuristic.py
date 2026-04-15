"""Deterministic heuristic baseline for backtesting.

Two modes:
  use_fixtures=True  (default): rank by form × (6 - fixture_difficulty)
  use_fixtures=False:           rank by form only

Both modes respect budget and team-limit constraints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeuristicResult:
    """Output of the heuristic baseline recommendation."""

    action: str  # "transfer" or "hold"
    player_out: dict | None
    player_in: dict | None
    captain_id: int
    captain_name: str
    reasoning: str


def _composite_score(player: dict, use_fixtures: bool) -> float:
    """Composite ranking score for a player.

    use_fixtures=True:  form × (6 - fixture_difficulty)
    use_fixtures=False: form only
    """
    form = player.get("form_float", 0.0)
    if not use_fixtures:
        return form
    difficulty = player.get("fixture", {}).get("difficulty", 3)
    return form * (6 - difficulty)


def _team_counts(squad: list[dict]) -> dict[int, int]:
    """Count players per PL team in a squad."""
    counts: dict[int, int] = {}
    for p in squad:
        counts[p["team"]] = counts.get(p["team"], 0) + 1
    return counts


def _post_transfer_team_ok(
    squad: list[dict],
    player_out: dict,
    player_in: dict,
) -> bool:
    """Check whether the in-player's team would stay within the 3-player limit
    after applying the transfer (out player's team decremented, in player added).
    """
    counts = _team_counts(squad)
    # Remove out player's contribution
    out_team = player_out["team"]
    counts[out_team] = counts.get(out_team, 1) - 1
    # Check in player
    in_team = player_in["team"]
    new_count = counts.get(in_team, 0) + 1
    return new_count <= 3


def heuristic_recommend(
    current_squad: list[dict],
    candidates: list[dict],
    bank: int,
    fixtures: list[dict],
    teams: dict[int, str],
    *,
    use_fixtures: bool = True,
) -> HeuristicResult:
    """Generate a deterministic transfer and captaincy recommendation.

    Transfer strategy:
      1. Find the outfield player (element_type != 1) with the lowest composite score.
      2. Among candidates at the same position: filter affordable + team-limit valid,
         rank by composite score descending, pick top.
      3. If no valid candidate, hold.

    Captain strategy:
      Highest composite score in the post-transfer squad (includes GKPs).

    Team-limit check uses post-transfer composition (out player's team
    decremented, in player's team incremented) to avoid a common off-by-one.
    """
    # Score outfield squad players (exclude GKPs from transfer-out)
    outfield = [p for p in current_squad if p.get("element_type", 0) != 1]
    if not outfield:
        # Fallback: nothing to transfer
        best_cap = max(current_squad, key=lambda p: _composite_score(p, use_fixtures))
        return HeuristicResult(
            action="hold",
            player_out=None,
            player_in=None,
            captain_id=best_cap["id"],
            captain_name=best_cap["web_name"],
            reasoning="No outfield players available for transfer. Hold.",
        )

    player_out = min(outfield, key=lambda p: _composite_score(p, use_fixtures))
    selling_price = player_out.get("selling_price", player_out.get("now_cost", 0))
    budget_for_in = bank + selling_price

    # Find best valid candidate at same position
    same_pos = [
        c for c in candidates
        if c.get("element_type") == player_out.get("element_type")
    ]

    player_in = None
    for candidate in sorted(same_pos, key=lambda c: _composite_score(c, use_fixtures), reverse=True):
        if candidate["now_cost"] > budget_for_in:
            continue
        if not _post_transfer_team_ok(current_squad, player_out, candidate):
            continue
        player_in = candidate
        break

    if player_in is None:
        # No valid candidate — hold, but still pick captain from current squad
        post_squad = current_squad
        best_cap = max(post_squad, key=lambda p: _composite_score(p, use_fixtures))
        return HeuristicResult(
            action="hold",
            player_out=None,
            player_in=None,
            captain_id=best_cap["id"],
            captain_name=best_cap["web_name"],
            reasoning=(
                f"No affordable valid replacement found for {player_out['web_name']} "
                f"(form {player_out['form_float']:.1f}). Hold. "
                f"Captain: {best_cap['web_name']}."
            ),
        )

    # Build post-transfer squad for captain selection
    post_squad = [p for p in current_squad if p["id"] != player_out["id"]] + [player_in]
    best_cap = max(post_squad, key=lambda p: _composite_score(p, use_fixtures))

    mode_label = "form×fixture" if use_fixtures else "form-only"
    out_fix = player_out.get("fixture", {})
    in_fix = player_in.get("fixture", {})
    reasoning = (
        f"[{mode_label}] Transfer: {player_out['web_name']} "
        f"(form {player_out['form_float']:.1f}, diff {out_fix.get('difficulty', '?')}) → "
        f"{player_in['web_name']} "
        f"(form {player_in['form_float']:.1f}, diff {in_fix.get('difficulty', '?')}). "
        f"Captain: {best_cap['web_name']} "
        f"(form {best_cap['form_float']:.1f}, diff {best_cap.get('fixture', {}).get('difficulty', '?')})."
    )

    return HeuristicResult(
        action="transfer",
        player_out=player_out,
        player_in=player_in,
        captain_id=best_cap["id"],
        captain_name=best_cap["web_name"],
        reasoning=reasoning,
    )
