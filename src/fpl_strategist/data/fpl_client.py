"""Async HTTP client for the FPL API with tenacity-based retry."""

from __future__ import annotations

import asyncio

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from fpl_strategist.data.models import (
    EntryInfo,
    Fixture,
    GameweekInfo,
    Pick,
    PicksResponse,
    Player,
    Team,
)

BASE_URL = "https://fantasy.premierleague.com/api"

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    reraise=True,
)


class FPLClient:
    """Async client for the Fantasy Premier League API."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._external_client = client is not None
        self._client = client or httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=15.0,
            headers={"User-Agent": "fpl-strategist/0.1.0"},
        )

    async def __aenter__(self) -> FPLClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._external_client:
            await self._client.aclose()

    @_RETRY
    async def _get(self, path: str) -> dict | list:
        resp = await self._client.get(path)
        resp.raise_for_status()
        return resp.json()

    async def get_bootstrap(self) -> dict:
        """GET /bootstrap-static/ — all players, teams, gameweeks."""
        return await self._get("/bootstrap-static/")

    async def get_entry(self, team_id: int) -> EntryInfo:
        """GET /entry/{team_id}/ — manager info."""
        data = await self._get(f"/entry/{team_id}/")
        return EntryInfo(**data)

    async def get_picks(self, team_id: int, gw: int) -> PicksResponse:
        """GET /entry/{team_id}/event/{gw}/picks/ — squad picks for a gameweek."""
        data = await self._get(f"/entry/{team_id}/event/{gw}/picks/")
        return PicksResponse(
            picks=[Pick(**p) for p in data["picks"]],
            entry_history=data["entry_history"],
        )

    async def get_fixtures(self, gw: int) -> list[Fixture]:
        """GET /fixtures/?event={gw} — fixtures for a gameweek."""
        data = await self._get(f"/fixtures/?event={gw}")
        return [Fixture(**f) for f in data]

    async def get_player_summary(self, player_id: int) -> dict:
        """GET /element-summary/{player_id}/ — per-player detailed stats."""
        return await self._get(f"/element-summary/{player_id}/")

    async def get_player_summaries(self, player_ids: list[int]) -> dict[int, dict]:
        """Fetch summaries for multiple players concurrently."""
        tasks = {pid: self.get_player_summary(pid) for pid in player_ids}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out: dict[int, dict] = {}
        for pid, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                continue
            out[pid] = result
        return out

    # --- Convenience methods that parse bootstrap data ---

    async def get_all_players(self) -> list[Player]:
        """Return all players from bootstrap-static."""
        bootstrap = await self.get_bootstrap()
        return [Player(**p) for p in bootstrap["elements"]]

    async def get_teams(self) -> list[Team]:
        """Return all teams from bootstrap-static."""
        bootstrap = await self.get_bootstrap()
        return [Team(**t) for t in bootstrap["teams"]]

    async def get_gameweeks(self) -> list[GameweekInfo]:
        """Return all gameweek info from bootstrap-static."""
        bootstrap = await self.get_bootstrap()
        return [GameweekInfo(**e) for e in bootstrap["events"]]

    async def get_next_gameweek(self) -> GameweekInfo | None:
        """Return the next unfinished gameweek, or None if season is over."""
        gws = await self.get_gameweeks()
        for gw in gws:
            if gw.is_next:
                return gw
        return None
