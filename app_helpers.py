"""Pure helper functions for building right-column Gradio components.

Each function takes the final graph state dict and returns display-ready output.
No LLM calls, no API calls, no side effects.
"""

from __future__ import annotations

POS_SORT = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
POS_LABELS = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}
STATUS_MAP = {"a": "\U0001f7e2", "d": "\U0001f7e1"}
FDR_COLORS = {
    1: "#00ff87",  # bright green
    2: "#00ff87",
    3: "#ebeb00",  # amber-yellow
    4: "#ff5a5a",  # red
    5: "#8b0000",  # dark red
}


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


def _fdr_badge(difficulty) -> str:
    """Inline colored badge for fixture difficulty rating."""
    try:
        diff = int(difficulty)
    except (ValueError, TypeError):
        return str(difficulty)
    color = FDR_COLORS.get(diff, "#aaa")
    text_color = "#000" if diff <= 3 else "#fff"
    return (
        f'<span style="background:{color};color:{text_color};'
        f'padding:1px 5px;border-radius:3px;font-size:0.8em;'
        f'font-weight:600;">{diff}</span>'
    )


def build_squad_html(state: dict) -> str:
    """Compact squad table grouped by position with fixture color coding."""
    squad, transferred_in_id = _build_post_transfer_squad(state)

    # Sort by position order, then alphabetically by name
    squad.sort(key=lambda p: (
        POS_SORT.get(p.get("position_name", "UNK"), 9),
        p["web_name"],
    ))

    rows: list[str] = []
    current_pos = None

    for p in squad:
        pos_order = POS_SORT.get(p.get("position_name", "UNK"), 9)
        if pos_order != current_pos:
            current_pos = pos_order
            label = POS_LABELS.get(pos_order, "OTHER")
            rows.append(
                f'<div style="font-size:0.75em;font-weight:700;opacity:0.5;'
                f'padding:6px 0 2px;text-transform:uppercase;letter-spacing:0.05em;">'
                f'{label}</div>'
            )

        name = p["web_name"]
        if transferred_in_id is not None and p["id"] == transferred_in_id:
            name = f"\U0001f504 {name}"

        fix = p.get("fixture", {})
        opp = fix.get("opponent", "?")
        loc = "H" if fix.get("is_home") else "A"
        diff = fix.get("difficulty", "?")

        status = STATUS_MAP.get(p.get("status", "a"), "\U0001f534")
        team = p.get("team_name", "?")
        form = p.get("form", "0.0")
        price = f"\u00a3{p['now_cost'] / 10:.1f}m"

        rows.append(
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'
            'font-size:0.9em;">'
            f'<span style="flex:1;min-width:0;">{status} <b>{name}</b>'
            f' <span style="opacity:0.5;">{team}</span></span>'
            f'<span style="width:45px;text-align:right;font-variant-numeric:tabular-nums;'
            f'opacity:0.7;margin-right:8px;">{form}</span>'
            f'<span style="width:55px;text-align:right;font-variant-numeric:tabular-nums;'
            f'margin-right:8px;">{price}</span>'
            f'<span style="width:70px;text-align:right;white-space:nowrap;">'
            f'{opp}({loc}) {_fdr_badge(diff)}</span>'
            '</div>'
        )

    return (
        '<div style="display:flex;flex-direction:column;">'
        + "".join(rows)
        + '</div>'
    )


def build_transfer_card(state: dict) -> str:
    """Markdown card for the transfer decision."""
    transfer = state.get("proposed_transfer")
    reasoning = state.get("transfer_reasoning", "")

    if transfer is None:
        md = "### Transfer\n**No transfer this gameweek.**"
        if reasoning:
            md += f"\n\n*{reasoning}*"
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
        md += f"\n\n*{reasoning}*"

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
        md += f"\n\n*{reasoning}*"

    return md


def build_summary_headline(state: dict) -> str:
    """Bold one-liner: Transfer: X -> Y | Captain: Z."""
    transfer = state.get("proposed_transfer")
    cap = state.get("captain_pick") or {}

    if transfer is None:
        transfer_str = "No transfer"
    else:
        transfer_str = f"{transfer['out']['web_name']} \u2192 {transfer['in']['web_name']}"

    cap_name = cap.get("name", "TBD")
    return f"**Transfer:** {transfer_str} **|** **Captain:** {cap_name} (C)"


def build_meta_footer(state: dict) -> str:
    """Compact one-line metadata footer."""
    gw = state.get("target_gw", "?")
    bank = state.get("bank", 0) / 10
    ft_used = 1 if state.get("proposed_transfer") is not None else 0
    replans = state.get("replan_count", 0)
    return f"GW {gw} \u00b7 Bank: \u00a3{bank:.1f}m \u00b7 FT used: {ft_used} \u00b7 Replans: {replans}"
