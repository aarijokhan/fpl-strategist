"""CLI entry point for the FPL Transfer Strategist."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fpl_strategist.data.fpl_client import FPLClient
from fpl_strategist.data.models import Player, SquadPlayer

app = typer.Typer(name="fpl", help="FPL Transfer Strategist — agentic transfer recommendations")
console = Console()

POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


async def _inspect(team_id: int, gw: int | None) -> None:
    async with FPLClient() as client:
        # Resolve gameweek
        bootstrap = await client.get_bootstrap()
        teams_lookup = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
        players_lookup = {p["id"]: p for p in bootstrap["elements"]}

        if gw is None:
            gw_info = await client.get_next_gameweek()
            if gw_info is None:
                console.print("[red]No upcoming gameweek found — season may be over.[/red]")
                raise typer.Exit(1)
            gw = gw_info.id
            console.print(f"Auto-detected next gameweek: [bold]GW {gw}[/bold]")

        # Fetch squad
        entry = await client.get_entry(team_id)
        picks_resp = await client.get_picks(team_id, gw - 1 if gw > 1 else gw)

        bank = picks_resp.entry_history.get("bank", 0)
        # Free transfers: 1 by default, 2 if they rolled over
        event_transfers = picks_resp.entry_history.get("event_transfers", 0)
        free_transfers = min(2, max(1, 2 - event_transfers)) if gw > 1 else 1

        # Fetch fixtures for target GW
        fixtures = await client.get_fixtures(gw)

        # Build fixture lookup: team_id -> "OPP (H/A) [difficulty]"
        fixture_display: dict[int, str] = {}
        for fix in fixtures:
            h_team = teams_lookup.get(fix.team_h, "???")
            a_team = teams_lookup.get(fix.team_a, "???")
            fixture_display[fix.team_h] = f"{a_team}(H) [{fix.team_h_difficulty}]"
            fixture_display[fix.team_a] = f"{h_team}(A) [{fix.team_a_difficulty}]"

        # Display header
        console.print(Panel(
            f"[bold]{entry.name}[/bold] — {entry.player_first_name} {entry.player_last_name}\n"
            f"Overall: {entry.summary_overall_points or '?'} pts | Rank: {entry.summary_overall_rank or '?'}",
            title=f"FPL Team #{team_id}",
        ))

        # Build squad table
        table = Table(title=f"Squad — GW {gw} Fixtures")
        table.add_column("Pos", style="cyan", width=4)
        table.add_column("Player", style="bold")
        table.add_column("Team", width=4)
        table.add_column("Form", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("EP Next", justify="right")
        table.add_column("Fixture", width=16)
        table.add_column("Status", width=6)
        table.add_column("Role", width=6)

        squad_players: list[dict] = []
        for pick in picks_resp.picks:
            p_data = players_lookup.get(pick.element, {})
            if not p_data:
                continue
            player = Player(**p_data)
            team_short = teams_lookup.get(player.team, "???")
            fix_str = fixture_display.get(player.team, "—")

            role = ""
            if pick.is_captain:
                role = "(C)"
            elif pick.is_vice_captain:
                role = "(V)"
            elif pick.multiplier == 0:
                role = "bench"

            status_style = ""
            if player.status == "i":
                status_style = "[red]INJ[/red]"
            elif player.status == "d":
                status_style = "[yellow]DBT[/yellow]"
            elif player.status == "s":
                status_style = "[red]SUS[/red]"
            else:
                status_style = "[green]AVL[/green]"

            table.add_row(
                POSITION_NAMES.get(player.element_type, "?"),
                player.web_name,
                team_short,
                player.form,
                f"£{player.now_cost / 10:.1f}m",
                str(player.ep_next or "—"),
                fix_str,
                status_style,
                role,
            )

        console.print(table)
        console.print(
            f"\n[bold]Budget:[/bold] £{bank / 10:.1f}m in bank | "
            f"[bold]Free transfers:[/bold] {free_transfers}"
        )


@app.command()
def inspect(
    team_id: int = typer.Argument(..., help="FPL team ID"),
    gw: int | None = typer.Option(None, "--gw", help="Target gameweek (auto-detects if omitted)"),
) -> None:
    """Fetch and display your FPL squad — no LLM calls, just data verification."""
    asyncio.run(_inspect(team_id, gw))


async def _recommend(team_id: int, gw: int | None, provider: str, verbose: bool) -> None:
    from dotenv import load_dotenv

    from fpl_strategist.graph import build_graph

    load_dotenv()

    # Auto-detect gameweek if needed
    if gw is None:
        async with FPLClient() as client:
            gw_info = await client.get_next_gameweek()
            if gw_info is None:
                console.print("[red]No upcoming gameweek found — season may be over.[/red]")
                raise typer.Exit(1)
            gw = gw_info.id
            console.print(f"Auto-detected next gameweek: [bold]GW {gw}[/bold]")

    graph = build_graph()
    initial_state = {
        "team_id": team_id,
        "target_gw": gw,
        "provider": provider,
    }

    console.print(f"\n[bold]Running agent for team {team_id}, GW {gw}...[/bold]\n")

    result = await graph.ainvoke(initial_state)

    # Verbose: dump key state fields
    if verbose:
        console.print("[dim]--- State dump ---[/dim]")
        for key in [
            "proposed_transfer", "transfer_reasoning", "is_valid", "violations",
            "replan_count", "captain_pick", "vice_captain_pick", "recommendation",
        ]:
            val = result.get(key)
            console.print(f"  [cyan]{key}:[/cyan] {val}")
        console.print()

    # Always show key outputs
    transfer = result.get("proposed_transfer")
    reasoning = result.get("transfer_reasoning", "")

    if transfer:
        out_p = transfer["out"]
        in_p = transfer["in"]
        console.print(f"[bold green]Transfer:[/bold green] {out_p['web_name']} → {in_p['web_name']}")
    else:
        console.print("[bold yellow]Transfer:[/bold yellow] Hold (no transfer)")

    console.print(f"[bold]Reasoning:[/bold] {reasoning}")

    recommendation = result.get("recommendation")
    if recommendation:
        console.print(f"\n[bold]Full recommendation:[/bold]\n{recommendation}")


@app.command()
def recommend(
    team_id: int = typer.Argument(..., help="FPL team ID"),
    gw: int | None = typer.Option(None, "--gw", help="Target gameweek"),
    provider: str = typer.Option("openai", "--provider", help="LLM provider: openai or anthropic"),
    verbose: bool = typer.Option(False, "--verbose", help="Show graph execution trace"),
) -> None:
    """Generate a transfer and captaincy recommendation for the next gameweek."""
    asyncio.run(_recommend(team_id, gw, provider, verbose))


@app.command()
def backtest(
    team_id: int = typer.Argument(..., help="FPL team ID"),
    from_gw: int = typer.Option(5, "--from-gw", help="Start gameweek"),
    to_gw: int = typer.Option(30, "--to-gw", help="End gameweek"),
) -> None:
    """Backtest the agent against historical gameweek outcomes."""
    console.print("[yellow]backtest command not yet implemented — coming in Phase 6[/yellow]")


if __name__ == "__main__":
    app()
