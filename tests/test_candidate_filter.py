"""Tests for the candidate filtering module."""

from __future__ import annotations

import pytest

from fpl_strategist.data.candidate_filter import filter_candidates, _candidate_score, _fixture_difficulty_bonus
from fpl_strategist.data.models import Fixture, Player


@pytest.fixture
def squad_ids():
    return {100, 101, 102}


@pytest.fixture
def fixtures():
    return [
        Fixture(id=1, event=10, team_h=1, team_a=2, team_h_difficulty=2, team_a_difficulty=4),
        Fixture(id=2, event=10, team_h=3, team_a=4, team_h_difficulty=3, team_a_difficulty=3),
    ]


@pytest.fixture
def player_pool(make_player):
    return [
        # Squad players — should be excluded
        make_player(id=100, web_name="InSquad1", element_type=3, team=1, form="7.0", ep_next="5.0"),
        make_player(id=101, web_name="InSquad2", element_type=2, team=2, form="6.0", ep_next="4.0"),
        make_player(id=102, web_name="InSquad3", element_type=4, team=3, form="5.0", ep_next="3.0"),
        # Available candidates — various positions
        make_player(id=1, web_name="HighFormMid", element_type=3, team=1, form="9.0", ep_next="7.0"),
        make_player(id=2, web_name="LowFormMid", element_type=3, team=2, form="1.0", ep_next="1.0"),
        make_player(id=3, web_name="HighFormDef", element_type=2, team=1, form="8.0", ep_next="5.0"),
        make_player(id=4, web_name="HighFormFwd", element_type=4, team=3, form="7.5", ep_next="6.0"),
        # Injured player — should NOT be filtered out (deliberately loose filtering)
        make_player(id=5, web_name="InjuredMid", element_type=3, team=4, form="8.0", ep_next="0.0", status="i",
                    news="Knee injury"),
    ]


def test_excludes_squad_players(player_pool, squad_ids, fixtures):
    candidates = filter_candidates(player_pool, squad_ids, fixtures)
    candidate_ids = {c.id for c in candidates}
    assert not candidate_ids & squad_ids


def test_does_not_filter_injured_players(player_pool, squad_ids, fixtures):
    """Deliberately loose: injured players are NOT filtered — validator catches them."""
    candidates = filter_candidates(player_pool, squad_ids, fixtures)
    candidate_ids = {c.id for c in candidates}
    assert 5 in candidate_ids


def test_ranks_by_form_and_ep(player_pool, squad_ids, fixtures):
    candidates = filter_candidates(player_pool, squad_ids, fixtures, per_position=2)
    mids = [c for c in candidates if c.element_type == 3]
    # HighFormMid (score: 9*2+7=25) should rank above InjuredMid (8*2+0=16) and LowFormMid (1*2+1=3)
    assert mids[0].web_name == "HighFormMid"


def test_limits_per_position(player_pool, squad_ids, fixtures):
    candidates = filter_candidates(player_pool, squad_ids, fixtures, per_position=1)
    mids = [c for c in candidates if c.element_type == 3]
    assert len(mids) == 1


def test_fixture_difficulty_bonus(make_player, fixtures):
    # Team 1 has home fixture with difficulty 2 -> bonus 1.0
    home_player = make_player(id=10, team=1, form="5.0", ep_next="3.0")
    bonus = _fixture_difficulty_bonus(home_player, fixtures, {})
    assert bonus == 1.0

    # Team 2 has away fixture with difficulty 4 -> no bonus
    away_hard = make_player(id=11, team=2, form="5.0", ep_next="3.0")
    bonus = _fixture_difficulty_bonus(away_hard, fixtures, {})
    assert bonus == 0.0

    # Team not in any fixture -> no bonus
    no_fix = make_player(id=12, team=99, form="5.0", ep_next="3.0")
    bonus = _fixture_difficulty_bonus(no_fix, fixtures, {})
    assert bonus == 0.0


def test_candidate_score(make_player, fixtures):
    player = make_player(id=1, team=1, form="5.0", ep_next="3.0")
    score = _candidate_score(player, fixtures, {})
    # 5.0*2 + 3.0 + 1.0 (home bonus) = 14.0
    assert score == 14.0


def test_empty_pool_returns_empty(squad_ids, fixtures):
    candidates = filter_candidates([], squad_ids, fixtures)
    assert candidates == []


def test_all_squad_returns_empty(make_player, fixtures):
    pool = [make_player(id=1, web_name="OnlyPlayer")]
    candidates = filter_candidates(pool, {1}, fixtures)
    assert candidates == []
