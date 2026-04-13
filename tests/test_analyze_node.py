"""Tests for the analyze_and_propose LLM node."""

from __future__ import annotations

import pytest

from fpl_strategist.nodes.analyze import analyze_and_propose


@pytest.mark.vcr("test_analyze_proposes_transfer.yaml")
async def test_analyze_proposes_transfer(sample_populated_state):
    """Invoke analyze_and_propose with a realistic state and verify output shape."""
    result = await analyze_and_propose(sample_populated_state)

    # Must return exactly these two keys
    assert "proposed_transfer" in result
    assert "transfer_reasoning" in result

    # Reasoning is always a non-empty string
    assert isinstance(result["transfer_reasoning"], str)
    assert len(result["transfer_reasoning"]) > 0

    # If a transfer was proposed, verify the structure
    if result["proposed_transfer"] is not None:
        transfer = result["proposed_transfer"]
        assert "out" in transfer
        assert "in" in transfer
        out_id = transfer["out"]["id"]
        in_id = transfer["in"]["id"]
        assert isinstance(out_id, int)
        assert isinstance(in_id, int)
        assert out_id != in_id
