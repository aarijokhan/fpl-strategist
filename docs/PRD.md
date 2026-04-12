# PRD: FPL Transfer Strategist

## Context

Portfolio project targeting AI engineer roles in UAE. The goal is to demonstrate genuine agentic architecture (planning, tool use, iterative replanning) — not an LLM wrapper. FPL is the domain because EPL has massive engagement in the UAE and is memorable at the portfolio stage.

The demo centerpiece is the **visible replan loop**: when the LLM proposes a transfer that violates deterministic constraints (budget, squad rules), the agent catches itself and replans. This proves the system is an agent, not a pipeline.

---

## Scope (MVP)

- **One transfer** recommendation for the **next gameweek**
- Captaincy pick (captain + vice-captain)
- Natural-language reasoning grounded in live FPL data
- Deterministic constraint validation (no LLM in the loop for rules)
- Evaluation via backtesting against historical gameweek outcomes
- CLI interface with Rich output + **Streamlit web UI** deployed to HF Spaces or Railway (live URL for recruiters)
- **Langfuse** integration for LLM tracing (screenshot in README)
- Target budgets: **<$0.05 per run**, **<15s p50 latency**

**Out of scope for MVP:** multi-gameweek lookahead, chip strategy, wildcard planning, multi-transfer planning.

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Orchestration | LangGraph >=1.1.6 | Agentic graph with visible replan loop |
| LLM | langchain-openai (GPT-4o default), langchain-anthropic (Sonnet as alt) | Structured output, cost-efficient |
| HTTP | httpx (async) + tenacity (retry) | Parallel fetches for 15 player summaries, resilient to transient failures |
| Tracing | Langfuse | LLM observability, cost tracking, screenshot for README |
| Web UI | Streamlit | Recruiter-facing live demo, deployed to HF Spaces / Railway |
| Data validation | Pydantic v2 | API response + LLM output parsing |
| CLI | Typer + Rich | Colored tables, progress spinners, panels |
| Testing | pytest + respx | Mock httpx without hitting live API |
| Eval | pandas (optional dep) | Backtest result analysis |
| Python | >=3.11 | |

---

## LangGraph Architecture

### State Schema (`state.py`)

```python
class FPLState(TypedDict):
    # Inputs
    team_id: int
    target_gw: int

    # Data layer (set by fetch_context)
    current_squad: list[dict]       # 15 players enriched with form, fixtures, price
    bank: int                       # Budget in tenths (15 = 1.5m)
    free_transfers: int             # 1 or 2
    all_players: list[dict]         # Pre-filtered ~60-80 transfer candidates
    fixtures: list[dict]            # Next GW fixtures with difficulty
    player_form: dict[int, dict]    # player_id -> recent stats

    # Proposal (set by analyze, overwritten by replan)
    proposed_transfer: dict | None  # {"out": player, "in": player} or None (hold)
    transfer_reasoning: str

    # Validation (set by validate)
    is_valid: bool
    violations: list[str]           # Human-readable constraint violations
    replan_count: int               # Caps replan loop at 2

    # Captain (set by select_captain)
    captain_pick: dict | None
    vice_captain_pick: dict | None

    # Final output
    recommendation: str

    # LLM trace
    messages: Annotated[list[BaseMessage], add_messages]
```

### Graph Topology

```
START → fetch_context → analyze_and_propose → validate_constraints
                                                    │
                                    ┌───────────────┼───────────────┐
                                    │               │               │
                              (valid=True)  (invalid, retries<2) (invalid, retries>=2)
                                    │               │               │
                                    │        replan_transfer        │
                                    │          (increments          │
                                    │          replan_count)   (fallback: hold)
                                    │               └──→ validate_constraints
                                    │
                                    └──→ select_captain → explain_recommendation → END
```

### Node Details

| Node | LLM? | What it does |
|------|-------|-------------|
| `fetch_context` | No | Calls FPL API, enriches squad data, passes candidates to filter module |
| `analyze_and_propose` | Yes | Evaluates squad weaknesses, proposes one transfer (or hold). Explicitly told NOT to check budget/rules — that's the validator's job |
| `validate_constraints` | **No** | Deterministic checks: budget, max 3 per team, same position, player availability. Returns violation list |
| `replan_transfer` | Yes | Increments `replan_count` explicitly, gets violation list injected into prompt, proposes a *different* transfer |
| `select_captain` | Yes | Picks captain/vice-captain from post-transfer squad based on fixture difficulty + form |
| `explain_recommendation` | Yes | Synthesizes everything into 3-5 paragraph natural-language recommendation |

### Conditional Routing

```python
def route_after_validation(state: FPLState) -> str:
    if state["is_valid"]:
        return "select_captain"
    if state["replan_count"] < 2:
        return "replan_transfer"
    # Exhausted replans → fall back to hold
    return "select_captain"
```

`replan_count` is incremented explicitly inside the `replan_transfer` node (not the router). When `replan_count >= 2`: clear `proposed_transfer` to None, set reasoning to "no valid transfer found within constraints — hold recommended."

---

## FPL API Data Layer

**Base URL:** `https://fantasy.premierleague.com/api/` (public, no auth)

### Endpoints used

| Endpoint | Data extracted | Used by |
|----------|---------------|---------|
| `/bootstrap-static/` | All players (elements), teams, gameweeks (events) | fetch_context |
| `/entry/{team_id}/` | Manager info, overall value | fetch_context |
| `/entry/{team_id}/event/{gw}/picks/` | Current squad picks, bank, transfers made | fetch_context |
| `/fixtures/?event={gw}` | Next GW fixtures + difficulty ratings | fetch_context |
| `/element-summary/{player_id}/` | Per-player history + upcoming fixtures (called for all 15 squad players) | fetch_context |

### Candidate Pre-filtering (`data/candidate_filter.py` — own tested module)

Deliberately loose so the constraint engine can reject proposals and fire the replan loop on real runs:

From ~700 players → ~60-80 candidates:
1. Exclude players already in squad
2. Per position, rank by: `float(form) * 2 + float(ep_next) + fixture_difficulty_bonus`
3. Take top 15-20 per position

**NOT filtered**: availability status, team counts. These are left for the constraint validator to catch, which is what triggers the visible replan loop in the demo.

### Known MVP Limitations

- **Selling price approximation**: Uses `now_cost` instead of exact selling price (which requires reconstructing full transfer history). Documented as known limitation.
- **Historical availability data**: `chance_of_playing` and `news` fields are point-in-time only — not available historically for backtesting. Acknowledged in eval docs.

---

## Constraint Engine (`constraints/engine.py`)

Entirely deterministic Python. Returns `ValidationResult(is_valid: bool, violations: list[str])`.

| Rule | Check | Violation message example |
|------|-------|--------------------------|
| Budget | `bank + sell_price - buy_price >= 0` | "Insufficient budget: need 9.2m, have 7.3m" |
| Team limit | No team with >3 players post-transfer | "MCI would have 4 players (max 3)" |
| Position match | Out and in must share `element_type` | "Cannot replace DEF with MID" |
| Availability | Incoming player `status` in `['a','d']` | "Isak unavailable: knee injury" |
| Squad membership | Out player in squad, in player exists | "Player not in your squad" |
| No self-swap | Out != In | "Cannot transfer a player for themselves" |

---

## LLM Prompts (purpose, not full text)

**analyze_and_propose**: "You're an FPL expert. Given this squad [table], form stats, fixtures, budget, and candidates [filtered list], propose ONE transfer or HOLD. Output structured JSON: `{action, player_out_id, player_in_id, reasoning}`. Do NOT enforce budget or squad rules."

**replan_transfer**: "Your proposal was rejected: {violations}. Same data. Propose a DIFFERENT transfer avoiding these constraints."

**select_captain**: "From this post-transfer squad with upcoming fixtures, pick captain and vice-captain. Output: `{captain_id, vice_captain_id, reasoning}`."

**explain_recommendation**: "Synthesize the transfer, captain pick, and reasoning into a 3-5 paragraph recommendation. Specific stats and fixtures. Conversational but authoritative."

All LLM nodes with structured output use `with_structured_output()` and Pydantic response models.

---

## Evaluation Framework

### Backtest approach

For each historical gameweek N in a range (e.g., GW 5-30):
1. Fetch the user's squad as of GW N-1
2. Run the agent (filtering player history to `round < N` to prevent data leakage)
3. Compare agent's recommendation against actual GW N outcomes

### Metrics

| Metric | Formula |
|--------|---------|
| Transfer delta | `points(transfer_in, GW N) - points(transfer_out, GW N)` |
| Captain delta | `agent_captain_points × 2 - user_captain_points × 2` |
| Transfer hit rate | % of GWs where transfer-in outscored transfer-out |
| Captain hit rate | % of GWs where agent's captain outscored user's |

### LLM-Judge Reasoning Coherence

In addition to point-based metrics, use an LLM-as-judge call to score the agent's reasoning:
- Input: the agent's `transfer_reasoning` + `captain_reasoning` + the factual context (form, fixtures)
- Scoring dimensions: factual grounding (does reasoning cite real stats?), logical coherence, actionability
- Output: 1-5 score per dimension
- Averaged across gameweeks and reported in README

### Baselines

- **User's actual decisions** (real transfers + captain)
- **Naive strategy** (no transfer, captain = highest EP player)
- **Heuristic baseline** (top-EP candidate at weakest position — deterministic, no LLM)

---

## CLI Interface

### Commands

```bash
fpl recommend 12345                    # Main command: one-GW recommendation
fpl recommend 12345 --verbose          # Show graph trace (demo mode)
fpl recommend 12345 --provider anthropic
fpl inspect 12345                      # Debug: show squad data, no LLM
fpl backtest 12345 --from-gw 5 --to-gw 30
```

### Output (recommend)

Rich-formatted panels showing:
1. Current squad table (player, team, form, price, fixture)
2. Budget + free transfers
3. Transfer recommendation with reasoning
4. **Replan note if constraint violation occurred** (the demo centerpiece)
5. Captain/vice-captain pick
6. Full natural-language explanation

### Verbose trace (--verbose)

```
[fetch_context]        ✓ Squad loaded: 15 players, bank 1.5m, 1 FT
[analyze_and_propose]  ✓ Proposed: Watkins → Haaland
[validate_constraints] ✗ VIOLATION: MCI would have 4 players (max 3)
[replan_transfer]      ✓ Revised: Watkins → Isak
[validate_constraints] ✓ All constraints passed
[select_captain]       ✓ Captain: Salah, Vice: Isak
[explain]              ✓ Done
```

---

## Project Structure

```
fpl-strategist/
├── pyproject.toml
├── .env.example                    # OPENAI_API_KEY=
├── .gitignore
├── README.md
├── src/fpl_strategist/
│   ├── __init__.py
│   ├── main.py                     # Typer CLI
│   ├── state.py                    # FPLState TypedDict
│   ├── graph.py                    # LangGraph graph definition
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── fetch_context.py
│   │   ├── analyze.py              # analyze_and_propose (LLM)
│   │   ├── validate.py             # validate_constraints (deterministic)
│   │   ├── replan.py               # replan_transfer (LLM)
│   │   ├── select_captain.py       # (LLM)
│   │   └── explain.py              # explain_recommendation (LLM)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fpl_client.py           # Async httpx client + tenacity retry
│   │   ├── models.py               # Pydantic models for API responses
│   │   └── candidate_filter.py     # Own tested module for candidate pre-filtering
│   ├── constraints/
│   │   ├── __init__.py
│   │   └── engine.py               # Deterministic constraint validator
│   └── eval/
│       ├── __init__.py
│       ├── backtest.py             # Backtesting harness
│       ├── scoring.py              # Point-outcome scoring
│       ├── heuristic.py            # Dumb heuristic baseline
│       └── judge.py                # LLM-judge reasoning coherence
├── streamlit_app.py                    # Streamlit web UI (top-level for deployment)
└── tests/
    ├── conftest.py
    ├── test_constraints.py
    ├── test_fpl_client.py
    ├── test_candidate_filter.py
    ├── test_graph.py
    └── test_eval.py
```

---

## Implementation Phases

### Phase 1: Scaffolding + Data Layer
- Project setup: `pyproject.toml`, directory structure, `.env.example`, `.gitignore`
- `fpl_client.py`: async httpx client with **tenacity-based retry** and all 5 endpoint methods
- `models.py`: Pydantic models for Player, SquadPlayer, Fixture
- `candidate_filter.py`: own tested module for pre-filtering candidates (form/fixtures only, no availability filter)
- `fpl inspect` CLI command to verify data fetching works
- Tests with `respx` mocks (including candidate filter tests)

### Phase 2: Constraint Engine
- `engine.py`: all 6 validation rules
- Thorough unit tests (most testable component)

### Phase 3: Graph Skeleton
- `state.py`: FPLState TypedDict
- `fetch_context` node (data fetching + candidate pre-filtering)
- `validate_constraints` node (calls constraint engine)
- Wire up `graph.py` with placeholder LLM nodes
- Test deterministic path end-to-end

### Phase 4: LLM Nodes
- `analyze_and_propose` with structured output
- `replan_transfer` with violation injection
- `select_captain` with structured output
- `explain_recommendation`
- Full graph integration test with real LLM

### Phase 5: CLI + Streamlit + Observability
- `fpl recommend` with Rich formatting
- `--verbose` trace mode
- **Streamlit web UI** (`streamlit_app.py`) wrapping the graph
- **Langfuse integration** for LLM tracing
- **Measure and report**: cost per run (<$0.05 target), p50 latency (<15s target)
- Error handling: team not found, API down, season ended
- Deploy to **HF Spaces or Railway** (live URL)

### Phase 6: Evaluation
- `scoring.py`: point-outcome metrics
- `heuristic.py`: dumb heuristic baseline (top-EP at weakest position)
- `judge.py`: LLM-judge reasoning coherence scoring (factual grounding, logic, actionability)
- `backtest.py`: harness with data leakage prevention
- Run backtests over 10-20 GWs

### Phase 7: README + Documentation
- Architecture diagram (Mermaid from LangGraph)
- Demo GIF of the Streamlit app
- Backtest results table with all metrics
- Langfuse trace screenshot
- Cost and latency measurements
- Known limitations and future work

---

## Verification

1. **Data layer**: `fpl inspect <real_team_id>` returns correctly formatted squad
2. **Constraints**: `pytest tests/test_constraints.py` — all rules tested with valid and invalid cases
3. **Happy path**: `fpl recommend <team_id>` produces a valid recommendation
4. **Replan path**: `fpl recommend <team_id> --verbose` on a squad where the top transfer candidate triggers a constraint violation → trace shows replan loop
5. **Eval**: `fpl backtest <team_id> --from-gw 10 --to-gw 15` produces a results table with metrics
6. **Graph visualization**: `graph.get_graph().draw_mermaid()` renders the topology with the visible replan cycle
