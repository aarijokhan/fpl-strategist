"""Deterministic constraint validation engine for FPL transfers.

Entirely rule-based — no LLM involvement. This is the safety net that
makes the replan loop meaningful: the LLM proposes, this engine validates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
MAX_PLAYERS_PER_TEAM = 3


@dataclass
class ValidationResult:
    is_valid: bool
    violations: list[str] = field(default_factory=list)


def validate_transfer(
    current_squad: list[dict],
    bank: int,
    proposed_out: dict | None,
    proposed_in: dict | None,
    all_players: list[dict] | None = None,
    teams: dict[int, str] | None = None,
) -> ValidationResult:
    """Validate a proposed transfer against FPL rules.

    Args:
        current_squad: List of squad player dicts, each with at minimum:
            id, web_name, element_type, team, now_cost, selling_price
        bank: Budget in bank (tenths, e.g. 15 = £1.5m)
        proposed_out: Player dict being transferred out (must be in squad), or None for hold
        proposed_in: Player dict being transferred in, or None for hold
        all_players: Full player pool (used for existence check)
        teams: team_id -> short_name mapping for readable violation messages

    Returns:
        ValidationResult with is_valid and list of violation strings.
    """
    # Hold transfer is always valid
    if proposed_out is None and proposed_in is None:
        return ValidationResult(is_valid=True)

    if teams is None:
        teams = {}

    violations: list[str] = []

    # Both must be specified for a transfer
    if proposed_out is None or proposed_in is None:
        violations.append("Transfer requires both a player out and a player in.")
        return ValidationResult(is_valid=False, violations=violations)

    # Rule 1: No self-swap
    if proposed_out["id"] == proposed_in["id"]:
        violations.append(
            f"Cannot transfer {proposed_out['web_name']} for themselves."
        )

    # Rule 2: Squad membership — out player must be in squad
    squad_ids = {p["id"] for p in current_squad}
    if proposed_out["id"] not in squad_ids:
        violations.append(
            f"{proposed_out['web_name']} (id={proposed_out['id']}) is not in your current squad."
        )

    # Rule 3: Player existence — in player must exist in pool (if pool provided)
    if all_players is not None:
        pool_ids = {p["id"] for p in all_players}
        if proposed_in["id"] not in pool_ids:
            violations.append(
                f"Player {proposed_in['web_name']} (id={proposed_in['id']}) not found in player pool."
            )

    # Rule 4: Position match
    out_pos = proposed_out.get("element_type")
    in_pos = proposed_in.get("element_type")
    if out_pos is not None and in_pos is not None and out_pos != in_pos:
        out_name = POSITION_NAMES.get(out_pos, str(out_pos))
        in_name = POSITION_NAMES.get(in_pos, str(in_pos))
        violations.append(
            f"Position mismatch: cannot replace {out_name} with {in_name}."
        )

    # Rule 5: Budget check
    sell_price = proposed_out.get("selling_price", proposed_out.get("now_cost", 0))
    buy_price = proposed_in.get("now_cost", 0)
    available = bank + sell_price
    if buy_price > available:
        violations.append(
            f"Insufficient budget: need £{buy_price / 10:.1f}m, "
            f"have £{available / 10:.1f}m "
            f"(bank £{bank / 10:.1f}m + sale £{sell_price / 10:.1f}m)."
        )

    # Rule 6: Team limit (max 3 per team)
    # Build post-transfer team counts
    team_counts: Counter[int] = Counter()
    for p in current_squad:
        if p["id"] != proposed_out["id"]:
            team_counts[p["team"]] += 1
    team_counts[proposed_in["team"]] += 1

    for team_id, count in team_counts.items():
        if count > MAX_PLAYERS_PER_TEAM:
            team_name = teams.get(team_id, f"team_{team_id}")
            violations.append(
                f"Team limit exceeded: {team_name} would have {count} players (max {MAX_PLAYERS_PER_TEAM})."
            )

    # Rule 7: Availability — incoming player status
    in_status = proposed_in.get("status", "a")
    if in_status not in ("a", "d"):
        status_labels = {"i": "injured", "s": "suspended", "u": "unavailable", "n": "not available"}
        label = status_labels.get(in_status, in_status)
        news = proposed_in.get("news", "")
        msg = f"{proposed_in['web_name']} is {label}"
        if news:
            msg += f": {news}"
        violations.append(msg + ".")

    return ValidationResult(
        is_valid=len(violations) == 0,
        violations=violations,
    )
