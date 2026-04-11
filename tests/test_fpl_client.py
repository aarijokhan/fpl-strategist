"""Tests for the FPL API client with mocked HTTP responses."""

from __future__ import annotations

import httpx
import pytest
import respx

from fpl_strategist.data.fpl_client import BASE_URL, FPLClient


@pytest.fixture
def mock_api():
    with respx.mock(base_url=BASE_URL) as api:
        yield api


@pytest.mark.asyncio
async def test_get_bootstrap(mock_api, sample_bootstrap_response):
    mock_api.get("/bootstrap-static/").respond(200, json=sample_bootstrap_response)

    async with FPLClient() as client:
        data = await client.get_bootstrap()

    assert len(data["elements"]) == 4
    assert data["elements"][0]["web_name"] == "Salah"


@pytest.mark.asyncio
async def test_get_entry(mock_api, sample_entry_response):
    mock_api.get("/entry/12345/").respond(200, json=sample_entry_response)

    async with FPLClient() as client:
        entry = await client.get_entry(12345)

    assert entry.id == 12345
    assert entry.name == "Test FC"


@pytest.mark.asyncio
async def test_get_picks(mock_api, sample_picks_response):
    mock_api.get("/entry/12345/event/9/picks/").respond(200, json=sample_picks_response)

    async with FPLClient() as client:
        picks = await client.get_picks(12345, 9)

    assert len(picks.picks) == 3
    assert picks.picks[0].is_captain is True
    assert picks.entry_history["bank"] == 15


@pytest.mark.asyncio
async def test_get_fixtures(mock_api, sample_fixtures_response):
    mock_api.get("/fixtures/").respond(200, json=sample_fixtures_response)

    async with FPLClient() as client:
        fixtures = await client.get_fixtures(10)

    assert len(fixtures) == 2
    assert fixtures[0].team_h == 10
    assert fixtures[0].team_h_difficulty == 2


@pytest.mark.asyncio
async def test_get_player_summary(mock_api):
    summary = {"history": [{"round": 1, "total_points": 8}], "fixtures": []}
    mock_api.get("/element-summary/1/").respond(200, json=summary)

    async with FPLClient() as client:
        data = await client.get_player_summary(1)

    assert data["history"][0]["total_points"] == 8


@pytest.mark.asyncio
async def test_get_player_summaries_concurrent(mock_api):
    for pid in [1, 2, 3]:
        mock_api.get(f"/element-summary/{pid}/").respond(
            200, json={"history": [{"round": 1, "total_points": pid * 10}], "fixtures": []}
        )

    async with FPLClient() as client:
        results = await client.get_player_summaries([1, 2, 3])

    assert len(results) == 3
    assert results[1]["history"][0]["total_points"] == 10
    assert results[3]["history"][0]["total_points"] == 30


@pytest.mark.asyncio
async def test_get_all_players(mock_api, sample_bootstrap_response):
    mock_api.get("/bootstrap-static/").respond(200, json=sample_bootstrap_response)

    async with FPLClient() as client:
        players = await client.get_all_players()

    assert len(players) == 4
    assert players[0].web_name == "Salah"
    assert players[0].form_float == 8.5


@pytest.mark.asyncio
async def test_get_next_gameweek(mock_api, sample_bootstrap_response):
    mock_api.get("/bootstrap-static/").respond(200, json=sample_bootstrap_response)

    async with FPLClient() as client:
        gw = await client.get_next_gameweek()

    assert gw is not None
    assert gw.id == 10
    assert gw.is_next is True


@pytest.mark.asyncio
async def test_retry_on_transient_error(mock_api, sample_entry_response):
    route = mock_api.get("/entry/12345/")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json=sample_entry_response),
    ]

    async with FPLClient() as client:
        entry = await client.get_entry(12345)

    assert entry.id == 12345


@pytest.mark.asyncio
async def test_raises_after_retries_exhausted(mock_api):
    mock_api.get("/entry/99999/").respond(500)

    async with FPLClient() as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_entry(99999)
