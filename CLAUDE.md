# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FPL Transfer Strategist is a portfolio project demonstrating genuine agentic AI architecture using LangGraph. It takes a user's Fantasy Premier League team ID and produces a one-gameweek transfer and captaincy recommendation with natural-language reasoning grounded in live data (fixtures, form, injuries, prices). The target audience is AI engineering recruiters in the UAE; the demo centerpiece is a visible replan loop where the agent proposes, fails constraint validation, and self-corrects. Full requirements, state schema, eval framework, and implementation phases are in [`docs/PRD.md`](docs/PRD.md).

## Architecture decisions

These are load-bearing — do not change without understanding the downstream effects:

- **Deterministic constraint node.** `validate_constraints` is pure Python, never LLM-based. This is what makes the replan loop meaningful — if the LLM validated its own proposals, violations would never surface. The engine checks 7 rules: budget, team limit (max 3), position match, availability, squad membership, player existence, no self-swap.
- **Replan loop capped at 2.** `replan_count` is incremented explicitly inside the `replan_transfer` node (not the router). After 2 failed attempts, the router clears the proposed transfer and proceeds directly to `select_captain` (there is no separate fallback node). This prevents infinite loops and keeps latency under the 15s target.
- **Candidate filter is loose on purpose.** `candidate_filter.py` only ranks by form/fixtures and excludes current squad. It does NOT filter by availability status or team counts. Those violations are left for the constraint engine to catch, which is what triggers the replan loop on real runs. Tightening the filter would make the replan loop unreachable.
- **Structured output via Pydantic.** All LLM nodes that return data use `with_structured_output()` with Pydantic response models (`TransferProposal`, `CaptainPick`). This ensures reliable JSON parsing and type safety.
- **No LLM in `fetch_context` or `validate_constraints`.** These nodes are deterministic. `fetch_context` calls the FPL API and runs the candidate filter. `validate_constraints` runs the constraint engine. Keeping them LLM-free makes the system testable, fast, and predictable.
- **Performance targets.** <$0.05 per run, <15s p50 latency. Measured and reported in Phase 5.

### LangGraph flow

```
fetch_context → analyze_and_propose (LLM) → validate_constraints (deterministic)
    → [valid] select_captain (LLM) → explain_recommendation (LLM) → END
    → [invalid, retries < 2] replan_transfer (LLM, increments replan_count) → validate_constraints
    → [invalid, retries >= 2] clear proposal, proceed to select_captain → explain_recommendation → END
```

## Out of scope

Multi-gameweek lookahead, chip strategy (wildcard/bench-boost/triple-captain/free-hit), multi-transfer planning, mini-league analysis, mobile app, Discord/Slack bot. If you find yourself building any of these, stop.

## File structure

```
app.py              # Gradio web UI (top-level for HF Spaces deployment)
app_helpers.py      # Gradio component builders (HTML cards, tables)
src/fpl_strategist/
├── data/           # FPL API client (async httpx + tenacity), Pydantic models, candidate filter
├── constraints/    # Deterministic transfer validation engine
├── nodes/          # One file per LangGraph node + schemas.py for Pydantic response models
├── eval/           # Backtesting harness, scoring, heuristic baseline, historical reconstruction, LLM-judge
├── main.py         # Typer CLI (inspect, recommend, backtest)
├── state.py        # LangGraph FPLState TypedDict
├── graph.py        # LangGraph graph definition with conditional routing
├── llm.py          # LLM factory with Langfuse callback wiring
tests/              # pytest + respx + VCR cassettes, one test file per module and node
docs/               # PRD.md, WALKTHROUGH.md, phase_5_baseline.md
```

## Conventions

- **Async httpx** for all FPL API calls, with tenacity retry (3 attempts, exponential backoff 0.5s-5s). Concurrent fetches for player summaries via `asyncio.gather`, capped with `asyncio.Semaphore` during backtests to avoid hammering the FPL API.
- **Pydantic v2** for all data models and LLM structured outputs. Player costs are in tenths (e.g., `100 = £10.0m`). Position codes use the FPL API's `element_type` field: 1=GKP, 2=DEF, 3=MID, 4=FWD.
- **`with_structured_output()`** on all LLM nodes that return structured data. Never parse JSON manually from LLM responses.
- **Imperative-mood lowercase commit messages** (e.g., "add constraint engine", "fix budget validation").
- **Tests required for every new module and every new node.** Test count must grow monotonically across phases — never remove passing tests. Use `respx` to mock httpx. Shared fixtures in `tests/conftest.py` (`make_player()`, `make_fixture()`, sample API responses). Async tests use pytest-asyncio with `asyncio_mode = "auto"`.

## Adding or modifying a node

1. Create or edit the node function in `src/fpl_strategist/nodes/<name>.py`. It receives `FPLState` and returns a partial state dict.
2. If the node uses an LLM, define a Pydantic response model and use `with_structured_output()`. If deterministic, no LLM import needed.
3. Register the node in `graph.py`: `builder.add_node("<name>", <function>)` and wire edges/conditional edges.
4. Add or update tests in `tests/test_<name>.py`. Mock LLM calls or use deterministic inputs. Verify the returned state dict keys and values.
5. Run the full suite (`python -m pytest tests/ -v`) and confirm test count has not decreased.

## Running locally

Package manager is **uv** (drop-in pip replacement, reads the same `pyproject.toml`). Lockfile is `uv.lock`.

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Required env vars (copy .env.example to .env)
# OPENAI_API_KEY        — required for recommend/backtest
# ANTHROPIC_API_KEY     — only if --provider anthropic
# LANGFUSE_PUBLIC_KEY   — for tracing (optional)
# LANGFUSE_SECRET_KEY   — for tracing (optional)

# Tests
python -m pytest tests/ -v                    # all tests
python -m pytest tests/test_constraints.py -v # single file
python -m pytest tests/test_constraints.py::TestBudget::test_over_budget_rejected -v  # single test

# Lint
ruff check src/ tests/

# Add a dependency
uv add <package>            # adds to pyproject.toml and uv.lock
uv add --dev <package>      # adds to dev optional-dependencies

# CLI
python -m fpl_strategist.main inspect 12345           # data verification, no LLM
python -m fpl_strategist.main recommend 12345 --verbose  # full agent run with trace
python -m fpl_strategist.main backtest 12345 --from-gw 5 --to-gw 30
```

CI runs on every push via `.github/workflows/test.yml` (Ubuntu, Python 3.11, `pytest`). If CI is red, fix it before moving on — do not stack changes on a broken main.

## Common pitfalls

- **Do NOT increment `replan_count` in the validator or the router.** It is incremented inside the `replan_transfer` node only. Moving it elsewhere breaks the loop count.
- **Do NOT add constraint enforcement to `analyze_and_propose`.** The LLM should focus on football intelligence. Budget, team limits, and availability are the validator's job. Mixing them defeats the replan loop.
- **Do NOT tighten the candidate filter** (e.g., by adding availability or team-count filters). The whole point is that invalid candidates reach the LLM, get proposed, fail validation, and trigger a visible replan. A tight filter makes the demo centerpiece unreachable.
- **Do NOT commit `.env`.** It contains API keys. `.gitignore` already excludes it.
- **`selling_price` is approximated as `now_cost`.** This is a known MVP limitation. Exact selling price requires reconstructing full transfer history from the API, which is out of scope.
