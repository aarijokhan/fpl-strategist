"""Pure helper functions for building right-column Gradio components.

Each function takes the final graph state dict and returns display-ready output.
No LLM calls, no API calls, no side effects.
"""

from __future__ import annotations

POS_SORT = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
STATUS_MAP = {"a": "\U0001f7e2", "d": "\U0001f7e1"}


def _build_post_transfer_squad(state: dict) -> tuple[list[dict], int | None]:
    """Apply the proposed transfer to the squad.

    Returns (post_transfer_squad, transferred_in_player_id_or_None).
    """
    squad = list(state.get("current_squad", []))
    transfer = state.get("proposed_transfer")
    if transfer is None:
        return squad, None
    out_id = transfer["out"]["id"]
    squad = [p for p in squad if p["id"] != out_id]
    in_player = transfer["in"]
    squad.append(in_player)
    return squad, in_player["id"]


def build_squad_html(state: dict) -> str:
    """HTML card list of the post-transfer squad sorted by position then name."""
    squad, transferred_in_id = _build_post_transfer_squad(state)

    # Sort by position order, then alphabetically by name
    squad.sort(key=lambda p: (
        POS_SORT.get(p.get("position_name", "UNK"), 9),
        p["web_name"],
    ))

    cards: list[str] = []
    for p in squad:
        name = p["web_name"]
        if transferred_in_id is not None and p["id"] == transferred_in_id:
            name = f"\U0001f504 {name}"

        fix = p.get("fixture", {})
        opp = fix.get("opponent", "?")
        loc = "H" if fix.get("is_home") else "A"
        diff = fix.get("difficulty", "?")
        next_str = f"{opp} ({loc}, {diff})"

        status = STATUS_MAP.get(p.get("status", "a"), "\U0001f534")
        pos = p.get("position_name", "UNK")
        team = p.get("team_name", "?")
        form = p.get("form", "0.0")
        price = f"\u00a3{p['now_cost'] / 10:.1f}m"

        cards.append(
            '<div style="padding:8px 12px;border-radius:6px;'
            'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);">'
            '<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span style="font-weight:600;">{status} {name}</span>'
            f'<span style="font-variant-numeric:tabular-nums;">{price}</span>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;font-size:0.85em;'
            f'opacity:0.7;margin-top:2px;">'
            f'<span>{pos} \u00b7 {team} \u00b7 form {form}</span>'
            f'<span>{next_str}</span>'
            '</div>'
            '</div>'
        )

    return (
        '<div style="display:flex;flex-direction:column;gap:6px;">'
        + "".join(cards)
        + '</div>'
    )


def build_transfer_card(state: dict) -> str:
    """Markdown card for the transfer decision."""
    transfer = state.get("proposed_transfer")
    reasoning = state.get("transfer_reasoning", "")

    if transfer is None:
        md = "### Transfer\n**No transfer this gameweek.**"
        if reasoning:
            md += f"\n\n> {reasoning}"
        return md

    out_p = transfer["out"]
    in_p = transfer["in"]
    out_pos = out_p.get("position_name", "?")
    in_pos = in_p.get("position_name", "?")
    out_team = out_p.get("team_name", "?")
    in_team = in_p.get("team_name", "?")

    net = (out_p["now_cost"] - in_p["now_cost"]) / 10
    net_str = f"+\u00a3{net:.1f}m to bank" if net >= 0 else f"-\u00a3{abs(net):.1f}m to bank"

    md = (
        f"### Transfer\n"
        f"**OUT:** {out_p['web_name']} ({out_pos}, {out_team}, "
        f"\u00a3{out_p['now_cost'] / 10:.1f}m, form {out_p['form']})\n"
        f"**IN:** {in_p['web_name']} ({in_pos}, {in_team}, "
        f"\u00a3{in_p['now_cost'] / 10:.1f}m, form {in_p['form']})\n"
        f"**Net:** {net_str}"
    )

    if reasoning:
        md += f"\n\n> {reasoning}"

    return md


def build_captain_card(state: dict) -> str:
    """Markdown card for captain and vice-captain selection."""
    cap = state.get("captain_pick") or {}
    vc = state.get("vice_captain_pick") or {}
    reasoning = state.get("captain_reasoning", "")

    # Look up full player info from the post-transfer squad
    squad, _ = _build_post_transfer_squad(state)
    squad_by_id = {p["id"]: p for p in squad}

    def _detail(pick: dict, player: dict) -> str:
        name = pick.get("name", "Unknown")
        team = player.get("team_name", "?")
        fix = player.get("fixture", {})
        opp = fix.get("opponent", "?")
        loc = "H" if fix.get("is_home") else "A"
        diff = fix.get("difficulty", "?")
        return f"{name} ({team}, vs {opp} {loc}, difficulty {diff})"

    cap_player = squad_by_id.get(cap.get("id"), {})
    vc_player = squad_by_id.get(vc.get("id"), {})

    md = (
        f"### Captaincy\n"
        f"**Captain:** {_detail(cap, cap_player)}\n"
        f"**Vice:** {_detail(vc, vc_player)}"
    )

    if reasoning:
        md += f"\n\n> {reasoning}"

    return md


def build_meta_footer(state: dict) -> str:
    """Compact one-line metadata footer."""
    gw = state.get("target_gw", "?")
    bank = state.get("bank", 0) / 10
    ft_used = 1 if state.get("proposed_transfer") is not None else 0
    replans = state.get("replan_count", 0)
    return f"GW {gw} \u00b7 Bank: \u00a3{bank:.1f}m \u00b7 FT used: {ft_used} \u00b7 Replans: {replans}"
