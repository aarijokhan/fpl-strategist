"""Pydantic response models for LLM structured output.

Used with `with_structured_output()` on all LLM nodes that return data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class TransferProposal(BaseModel):
    """Structured output from analyze_and_propose and replan_transfer nodes."""

    action: Literal["transfer", "hold"]
    player_out_id: int | None = None
    player_in_id: int | None = None
    reasoning: str


class CaptainPick(BaseModel):
    """Structured output from select_captain node."""

    captain_id: int
    vice_captain_id: int
    reasoning: str

    @model_validator(mode="after")
    def captain_and_vice_must_differ(self) -> CaptainPick:
        if self.captain_id == self.vice_captain_id:
            raise ValueError("Captain and vice-captain must be different players.")
        return self
