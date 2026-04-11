"""Pydantic models for FPL API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Player(BaseModel):
    """A player from the bootstrap-static elements array."""

    id: int
    web_name: str
    element_type: int  # 1=GKP, 2=DEF, 3=MID, 4=FWD
    team: int
    now_cost: int  # In tenths, e.g. 100 = £10.0m
    form: str  # String from API, parse to float for sorting
    total_points: int
    ep_next: str | None = None  # Expected points next GW
    expected_goals: str = "0.00"
    expected_assists: str = "0.00"
    minutes: int = 0
    news: str = ""
    chance_of_playing_next_round: int | None = None
    status: str = "a"  # 'a'=available, 'd'=doubtful, 'i'=injured, 's'=suspended, 'u'=unavailable

    @property
    def form_float(self) -> float:
        try:
            return float(self.form)
        except (ValueError, TypeError):
            return 0.0

    @property
    def ep_next_float(self) -> float:
        try:
            return float(self.ep_next) if self.ep_next else 0.0
        except (ValueError, TypeError):
            return 0.0

    @property
    def position_name(self) -> str:
        return {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(self.element_type, "UNK")


class SquadPlayer(Player):
    """A player in the user's current squad, enriched with squad-specific data."""

    selling_price: int  # Approximate selling price in tenths
    is_captain: bool = False
    is_vice_captain: bool = False
    multiplier: int = 1  # 0=bench, 1=playing, 2=captain, 3=triple-captain


class Team(BaseModel):
    """A Premier League team."""

    id: int
    name: str
    short_name: str


class Fixture(BaseModel):
    """A single fixture for a gameweek."""

    id: int
    event: int  # Gameweek number
    team_h: int
    team_a: int
    team_h_difficulty: int = Field(ge=1, le=5)
    team_a_difficulty: int = Field(ge=1, le=5)
    finished: bool = False
    team_h_score: int | None = None
    team_a_score: int | None = None


class GameweekInfo(BaseModel):
    """Metadata about a gameweek from the events array."""

    id: int
    name: str
    deadline_time: str
    is_current: bool
    is_next: bool
    finished: bool


class EntryInfo(BaseModel):
    """Manager entry info from /entry/{team_id}/."""

    id: int
    player_first_name: str = ""
    player_last_name: str = ""
    name: str  # Team name
    summary_overall_points: int | None = None
    summary_overall_rank: int | None = None


class Pick(BaseModel):
    """A single pick from the picks endpoint."""

    element: int  # Player ID
    position: int  # Squad position 1-15
    multiplier: int  # 0=bench, 1=playing, 2=captain, 3=TC
    is_captain: bool
    is_vice_captain: bool


class PicksResponse(BaseModel):
    """Response from /entry/{team_id}/event/{gw}/picks/."""

    picks: list[Pick]
    entry_history: dict  # Contains bank, value, event_transfers, etc.
