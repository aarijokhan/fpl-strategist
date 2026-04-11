"""Candidate pre-filtering module.

Deliberately loose: filters by form and fixture difficulty only.
Does NOT filter by availability status or team counts — those are left
for the constraint engine to catch, which triggers the replan loop.
"""

from __future__ import annotations

from fpl_strategist.data.models import Fixture, Player

# Position element_type codes
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# How many candidates to keep per position
CANDIDATES_PER_POSITION = 20


def _fixture_difficulty_bonus(player: Player, fixtures: list[Fixture], teams: dict[int, str]) -> float:
    """Give a bonus for players whose team has a favourable home fixture."""
    for fix in fixtures:
        if fix.team_h == player.team and fix.team_h_difficulty <= 2:
            return 1.0
        if fix.team_a == player.team and fix.team_a_difficulty <= 2:
            return 0.5
    return 0.0


def _candidate_score(player: Player, fixtures: list[Fixture], teams: dict[int, str]) -> float:
    """Composite ranking score: form-weighted + expected points + fixture bonus."""
    return (
        player.form_float * 2.0
        + player.ep_next_float
        + _fixture_difficulty_bonus(player, fixtures, teams)
    )


def filter_candidates(
    all_players: list[Player],
    squad_ids: set[int],
    fixtures: list[Fixture],
    teams: dict[int, str] | None = None,
    per_position: int = CANDIDATES_PER_POSITION,
) -> list[Player]:
    """Filter the full player pool to a manageable set of transfer candidates.

    Filtering rules (deliberately loose):
    1. Exclude players already in the squad.
    2. Per position, rank by composite score (form + ep_next + fixture bonus).
    3. Take top `per_position` per position.

    NOT filtered: availability status, team counts. These are left for the
    constraint validator so the replan loop fires on real runs.
    """
    if teams is None:
        teams = {}

    # Exclude current squad
    pool = [p for p in all_players if p.id not in squad_ids]

    # Group by position and rank
    by_position: dict[int, list[Player]] = {pos: [] for pos in POSITIONS}
    for player in pool:
        if player.element_type in by_position:
            by_position[player.element_type].append(player)

    candidates: list[Player] = []
    for pos, players in by_position.items():
        ranked = sorted(
            players,
            key=lambda p: _candidate_score(p, fixtures, teams),
            reverse=True,
        )
        candidates.extend(ranked[:per_position])

    return candidates
