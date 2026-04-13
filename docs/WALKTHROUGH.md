# Building an FPL Transfer Strategist: A Walkthrough

> **Style guidance:** Written in the style of Kernighan and Ritchie — short sentences, concrete examples, minimal jargon, one concept per paragraph.

---

## 1. Introduction

This project recommends one Fantasy Premier League transfer per gameweek. It fetches live data from the FPL API, asks an LLM to propose a transfer, validates the proposal against deterministic rules, and — if the proposal breaks a rule — loops back and asks the LLM to try again. That loop is the whole point. Without it, you have a wrapper around ChatGPT. With it, you have an agent.

The domain is FPL because the rules are public, the data is free, and millions of people in the UAE care about the Premier League. But the interesting part is not the domain. The interesting part is that the system catches its own mistakes and corrects them without human intervention.

The project is built in six phases. Each phase adds one capability and is independently testable. This walkthrough covers the first three: fetching data, validating constraints, and wiring the graph. The LLM nodes, the UI, and the evaluation framework come later.

---

## 2. Mental Model

### 2.1 What is an agent vs. a script?

A script runs top to bottom. It calls an API, gets a response, prints it. If the response is wrong, the script does not know.

An agent has a feedback loop. It proposes an action, checks whether the action is valid, and — if not — revises the proposal. The check must be independent of the proposal. In this project, the LLM proposes transfers and a deterministic constraint engine validates them. The LLM never sees the constraint code. The constraint engine never calls the LLM. They communicate through shared state.

Here is the router that closes the loop:

```python
def _route_after_validation(state: FPLState) -> str:
    if state.get("is_valid", False):
        return "select_captain"
    if state.get("replan_count", 0) < MAX_REPLANS:
        return "replan_transfer"
    return "select_captain"
```

Three exits: valid, invalid with retries left, invalid with retries exhausted. That conditional edge is what makes this an agent.

### 2.2 What is a state machine?

A state machine is a system that is always in exactly one state, and moves to the next state based on rules. In LangGraph, the states are nodes — Python functions — and the rules are edges. Shared data lives in a `TypedDict` that every node can read and write.

```python
class FPLState(TypedDict, total=False):
    team_id: int
    target_gw: int
    current_squad: list[dict]
    bank: int
    proposed_transfer: dict | None
    is_valid: bool
    violations: list[str]
    replan_count: int
    # ... more fields
```

`total=False` means no field is required at construction time. The graph starts with just `team_id` and `target_gw`. Each node adds its own fields. By the time `explain_recommendation` runs, every field is populated.

The graph never holds mutable global state. Everything is in the `TypedDict`, passed node to node. This makes testing trivial: construct a dict, call a node function, assert on the returned dict.

---

## 3. Phase 1: Scaffolding + Data Layer

### 3.1 What I built

`data/fpl_client.py` — async HTTP client with retry logic for five FPL API endpoints. `data/models.py` — Pydantic models for every API response shape: `Player`, `Fixture`, `Pick`, `PicksResponse`, and more. `data/candidate_filter.py` — pre-filtering module that narrows ~700 players down to ~80 transfer candidates. `main.py` — a Typer CLI with an `inspect` command that dumps your squad to the terminal.

### 3.2 Why it exists

If you skip the data layer and go straight to the LLM, you have no ground truth. The LLM will hallucinate player prices, invent fixtures, and propose transfers for players who do not exist. Every claim the LLM makes must be checkable against real data. Phase 1 ensures that data exists before Phase 4 asks the LLM to reason about it.

### 3.3 Key concepts

**Async HTTP with retry.** The FPL API is public and unauthed, which means it rate-limits aggressively. The client uses tenacity to retry transient failures with exponential backoff:

```python
_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    retry=retry_if_exception_type(
        (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
    ),
    reraise=True,
)
```

Every `_get` call is decorated with `_RETRY`. After three failures, the exception propagates. This keeps the client honest: it retries on flakes, but does not mask real errors.

**Pydantic models with computed properties.** The FPL API returns player form and expected points as strings. The models parse them once:

```python
class Player(BaseModel):
    form: str       # "4.7" from the API
    ep_next: str | None = None

    @property
    def form_float(self) -> float:
        try:
            return float(self.form)
        except (ValueError, TypeError):
            return 0.0
```

Downstream code calls `player.form_float` instead of scattering `float()` conversions everywhere. The parsing logic lives in one place.

**Deliberately loose candidate filtering.** The filter excludes squad players and ranks by a composite score, but it does not check availability or team limits:

```python
pool = [p for p in all_players if p.id not in squad_ids]
```

This is intentional. An injured player or a fourth player from the same team will pass through the filter and into the LLM's candidate list. The constraint engine in Phase 2 catches it. The LLM gets told "no," and the replan loop fires. If we filtered perfectly here, the replan loop would never trigger, and the demo would not demonstrate agency.

### 3.4 What confused me

Player costs are stored in tenths. A player priced at £10.0m has `now_cost = 100`. I wrote the first version of the budget check using pounds and spent an hour wondering why every transfer was affordable. The FPL API documentation does not mention the unit. I found it by comparing `now_cost` values to the website.

Selling price is not the same as buying price. FPL uses a profit-sharing formula for selling. The exact selling price requires reconstructing the full transfer history, which is beyond the MVP. The client uses `now_cost` as an approximation. This is a known inaccuracy, documented and accepted.

---

## 4. Phase 2: Constraint Engine

### 4.1 What I built

`constraints/engine.py` — a single function, `validate_transfer`, that checks seven rules and returns a `ValidationResult` with a boolean and a list of human-readable violation strings.

### 4.2 Why it exists

The LLM cannot be trusted to follow rules. It will propose a transfer that busts the budget, or brings in a fourth player from Manchester City, or swaps a defender for a midfielder. These are not edge cases — they are the common case when you give an LLM a list of 80 candidates and say "pick one."

The constraint engine is entirely deterministic. No LLM, no probability, no "usually works." It either passes or it does not. This is the safety net that makes the replan loop meaningful: if the validator were also an LLM, the system would be two LLMs arguing about rules neither understands.

### 4.3 Key concepts

**Violations as human-readable strings.** Each rule produces a message that the replan node can inject directly into the LLM prompt:

```python
violations.append(
    f"Insufficient budget: need £{buy_price / 10:.1f}m, "
    f"have £{available / 10:.1f}m "
    f"(bank £{bank / 10:.1f}m + sale £{sell_price / 10:.1f}m)."
)
```

The LLM does not need to interpret error codes. It reads "Insufficient budget: need £9.2m, have £7.3m" and proposes a cheaper player. The violation messages are the communication channel between the deterministic engine and the LLM.

**Post-transfer team counts.** The team-limit check does not look at the current squad. It simulates the squad after the transfer and counts:

```python
team_counts: Counter[int] = Counter()
for p in current_squad:
    if p["id"] != proposed_out["id"]:
        team_counts[p["team"]] += 1
team_counts[proposed_in["team"]] += 1
```

This catches the case where transferring out a player from one team and bringing in a player from another still violates the three-per-team cap.

**Hold transfers are always valid.** If both `proposed_out` and `proposed_in` are `None`, the engine returns `is_valid=True` immediately. This matters because the fallback after exhausted replans is to hold — and a hold must never fail validation.

### 4.4 What confused me

I initially wrote the position check as `proposed_out["element_type"] == proposed_in["element_type"]` without guarding against missing keys. The LLM response schemas had not been built yet, and during testing I was passing hand-crafted dicts that sometimes omitted `element_type`. The validator would silently skip the check. I added `.get()` with explicit `None` handling so a missing field never masks a real violation.

---

## 5. Phase 3: Graph Skeleton

### 5.1 What I built

`state.py` — the `FPLState` TypedDict shared by all nodes. `graph.py` — the LangGraph `StateGraph` with six nodes, conditional routing, and the replan loop. `nodes/fetch_context.py` — the data-fetching node that populates the state. `nodes/validate.py` — the thin wrapper that calls the constraint engine. `nodes/replan.py`, `nodes/analyze.py`, `nodes/select_captain.py`, `nodes/explain.py` — placeholders that return minimal state updates so the graph can compile and the deterministic path can be tested end-to-end.

### 5.2 Why it exists

If you build LLM nodes first and graph structure second, you cannot test the architecture without spending money on API calls. Phase 3 builds the graph with placeholder nodes that return canned data. The routing logic, the replan loop, and the fallback behavior are all testable without an LLM key. By the time Phase 4 plugs in real LLM calls, the graph structure is already proven correct.

### 5.3 Key concepts

**Conditional edges and the replan loop.** The graph is mostly linear: fetch, analyze, validate. The interesting part is the conditional edge after validation:

```python
builder.add_conditional_edges(
    "validate_constraints",
    _route_after_validation,
    {
        "select_captain": "select_captain",
        "replan_transfer": "replan_transfer",
    },
)
builder.add_edge("replan_transfer", "validate_constraints")
```

`replan_transfer` always loops back to `validate_constraints`. The only exit from this cycle is the router deciding the proposal is valid or the retries are exhausted.

**replan_count is incremented in exactly one place.** The replan node increments it. Not the router. Not the validator. This invariant is critical:

```python
async def replan_transfer(state: FPLState) -> dict:
    new_count = state.get("replan_count", 0) + 1
    return {
        "replan_count": new_count,
        "proposed_transfer": None,
        # ...
    }
```

If the router also incremented the count, a single failed proposal could burn two retries. The tests assert this: route the graph through two replan cycles and verify `replan_count == 2`, not 3 or 4.

**Fallback clears the proposal.** When replans are exhausted and the proposal is still invalid, the graph does not crash. It clears the proposal and proceeds to captain selection:

```python
def _clear_proposal_if_exhausted(state: FPLState) -> dict:
    if not state.get("is_valid", False) and state.get("replan_count", 0) >= MAX_REPLANS:
        return {
            "proposed_transfer": None,
            "transfer_reasoning": (
                "After multiple attempts, no valid transfer could be found "
                "within constraints. Recommendation: hold the free transfer."
            ),
        }
    return {}
```

This means the graph always produces output. A failed replan does not leave the user staring at an error — it produces a "hold" recommendation with an honest explanation.

### 5.4 What confused me

`total=False` on the `TypedDict` was non-obvious. LangGraph expects nodes to return partial dicts — just the fields they changed. With `total=True` (the default), type checkers complain about missing keys in those partial returns. Setting `total=False` makes every field optional at the type level, which matches how the graph actually uses the state: sparse at the start, fully populated by the end.

I also expected LangGraph's `add_conditional_edges` to accept a simple function returning a node name. It does, but it also requires a mapping dict from return values to node names. The mapping is redundant when the return values are already node names, but LangGraph uses it for graph visualization — it needs to know the possible targets at compile time, not just at runtime.

---

## 6. Phase 4: LLM Nodes

### 6.1 What I built

> TODO: write this section.

### 6.2 Why it exists

> TODO: write this section.

### 6.3 Key concepts

> TODO: write this section.

### 6.4 What confused me

> TODO: write this section.

---

## 7. Phase 5: CLI + Streamlit + Observability

### 7.1 What I built

> TODO: write this section.

### 7.2 Why it exists

> TODO: write this section.

### 7.3 Key concepts

> TODO: write this section.

### 7.4 What confused me

> TODO: write this section.

---

## 8. Phase 6: Evaluation

### 8.1 What I built

> TODO: write this section.

### 8.2 Why it exists

> TODO: write this section.

### 8.3 Key concepts

> TODO: write this section.

### 8.4 What confused me

> TODO: write this section.

---

## 9. Glossary

> TODO: write this section.

---

## 10. What I'd Do Differently

> TODO: write this section.
