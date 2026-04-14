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

Fixture difficulty ratings come from the official FPL API, not from a model I built. They are set by the Premier League's FPL team based on overall team strength and home/away advantage. They do not update dynamically based on recent form. A team on a ten-game losing streak still carries the same FDR it was assigned at calibration time. The UI displays these ratings with color coding (green for easy, red for hard), but the underlying numbers are upstream data, not a prediction. A future improvement could replace FDR with a custom difficulty score derived from rolling xG data.

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

`nodes/schemas.py` — two Pydantic models, `TransferProposal` and `CaptainPick`, used as structured output schemas for every LLM node that returns data. `nodes/analyze.py` — the first LLM node, which evaluates the squad and proposes one transfer. `nodes/replan.py` — the correction node, which sees the violation list and proposes a different transfer. `nodes/select_captain.py` — picks captain and vice-captain from the post-transfer squad. `nodes/explain.py` — synthesizes everything into a natural-language recommendation. `llm.py` — factory function that returns a configured `ChatOpenAI` or `ChatAnthropic` with optional Langfuse tracing.

### 6.2 Why it exists

Phase 3 proved the graph works with canned data. Phase 4 replaces the canned data with real LLM calls. The separation matters because it means Phase 4 only had to get the prompts right — the routing, the replan loop, and the fallback behavior were already tested.

### 6.3 Key concepts

**Structured output via Pydantic.** Every LLM node that returns data uses `with_structured_output()` with a Pydantic model:

```python
structured_llm = llm.with_structured_output(TransferProposal)
result: TransferProposal = await structured_llm.ainvoke([HumanMessage(content=prompt)])
```

The LLM is instructed to return JSON matching the schema. LangChain parses and validates the response automatically. If the LLM returns malformed JSON, the structured output wrapper retries. This eliminates manual JSON parsing.

`CaptainPick` goes further with a model-level validator:

```python
class CaptainPick(BaseModel):
    captain_id: int
    vice_captain_id: int
    reasoning: str

    @model_validator(mode="after")
    def captain_and_vice_must_differ(self) -> CaptainPick:
        if self.captain_id == self.vice_captain_id:
            raise ValueError("Captain and vice-captain must be different players.")
        return self
```

If the LLM returns the same player for both captain and vice-captain, Pydantic rejects the response before the node ever sees it.

**Prompt responsibility boundaries.** The analyze prompt explicitly disclaims responsibility for constraints:

```
Do NOT check or enforce any of the following — they are validated
deterministically downstream and are NOT your concern:
- Budget constraints
- Team limits (max 3 players per Premier League team)
- Squad size or formation rules
- Player availability, injury status, or suspensions
```

This is not politeness. If the LLM tries to enforce budget rules, it will reject good candidates prematurely and the replan loop will never fire. The whole architecture depends on the LLM proposing on footballing merit alone and the deterministic validator catching mistakes.

**Violation injection in the replan prompt.** The replan node receives the exact violation strings from the constraint engine and injects them into the prompt:

```
# Violations
The constraint validator rejected your proposal for these reasons:
1. Insufficient budget: need £8.8m, have £6.8m (bank £1.2m + sale £5.6m).

# Task
Propose a DIFFERENT transfer that avoids the violations listed above.
```

The LLM reads the violation, understands it needs a cheaper player, and proposes one. The violation messages are the communication channel between the deterministic engine and the LLM.

**Rejected proposal history.** The replan node does not just fix the current failure. It also records what failed and why:

```python
rejected_proposals.append(prev_proposal)
replan_violations_history.append(list(state.get("violations", [])))
```

This history is passed to the explain node, which weaves it into the recommendation: "We initially considered Gyökeres, but that was off the table due to budget constraints." The user sees the full decision-making process, not just the final answer.

**Two model tiers.** The analyze, replan, and captain nodes use `gpt-4o-mini` — fast, cheap, sufficient for structured proposals. The explain node uses `gpt-4o` — the larger model is justified because the final prose is the user-facing output and benefits from better writing. This keeps the per-run cost under $0.05 while the recommendation reads well.

**Defensive ID resolution.** Even with structured output, the LLM can hallucinate a player ID that does not exist in the candidate list. Every node checks:

```python
out_player = squad_by_id.get(result.player_out_id)
in_player = candidates_by_id.get(result.player_in_id)

if out_player is None or in_player is None:
    return {"proposed_transfer": None, "transfer_reasoning": "...could not be resolved..."}
```

An unresolvable ID degrades gracefully to a hold recommendation rather than crashing the graph.

**Force-replan demo mode.** When `force_replan=True`, the analyze node skips the LLM entirely and returns a deliberately unaffordable transfer — the cheapest squad player out, the most expensive candidate in:

```python
most_expensive = max(candidates, key=lambda c: c["now_cost"])
cheapest = min(same_pos, key=lambda p: p.get("selling_price", p["now_cost"]))
```

This guarantees a budget violation, which triggers the replan loop, which produces the project's best demo moment — the Gyökeres → Beto recovery — without spending on an LLM call for the first proposal.

### 6.4 What confused me

The explain node does not use `with_structured_output()`. The other three LLM nodes do. I initially added it to explain as well, wrapping the prose in a `{"recommendation": "..."}` JSON object. This made the LLM write worse prose — it would truncate paragraphs to fit the JSON structure and add escaping artifacts. Removing structured output and reading `response.content` directly fixed the quality immediately. Structured output is for data. Prose is not data.

I also underestimated how important the prompt's negative instructions are. "Do NOT check budget constraints" in the analyze prompt is more valuable than "DO propose the best transfer." Without the negative instruction, the LLM would reject half the candidates on its own, making the constraint engine redundant and the replan loop unreachable.

---

## 7. Phase 5: Web UI + Resilience

### 7.1 What I built

`app.py` — a Gradio web UI that streams the graph execution as collapsible ChatMessage cards. `app_helpers.py` — pure functions that build the right-column components: squad HTML, transfer card, captain card, summary headline, meta footer. `llm.py` gained a BYOK (Bring Your Own Key) mechanism. `data/demo_cache.json` — a static cached trace for graceful fallback when the FPL API is down.

The CLI (`main.py`) was built in earlier phases. It has three commands: `inspect` (data only, no LLM), `recommend` (full graph run with optional verbose trace), and `backtest` (stubbed for Phase 6).

### 7.2 Why it exists

A CLI is fine for development. A recruiter will not run a CLI. The web UI exists so someone can click a button, watch the agent reason in real time, and see the replan loop fire — all without installing anything. The streaming trace is the demo centerpiece: each node appears as a card that transitions from "pending" to "done," and the failed validation → replan sequence is visible as it happens.

The resilience layer exists because the FPL API is unreliable. It goes down during matches, returns 404 for invalid team IDs, and shuts off entirely between seasons. A portfolio demo that shows a blank error page when a recruiter visits is worse than no demo at all.

### 7.3 Key concepts

**Streaming with `graph.astream()`.** The Gradio handler is an async generator that yields after every node completes:

```python
async for event in graph.astream(initial_state):
    for node_name, update in event.items():
        cumulative_state.update(update)
        messages.append(ChatMessage(
            role="assistant",
            content=content,
            metadata={"title": title, "status": "pending"},
        ))
        yield messages, _S, recommendation, _S, _S, _S, _S
```

Each yield updates the chatbot. The `_S` values are `gr.skip()` — they tell Gradio "do not touch this component." Without them, the right-column components flash loading indicators on every yield. On the final yield, all components populate at once.

**Error taxonomy.** Not all errors are the same, and the UI treats them differently:

- 404 (team not found) → "Team ID not found" with instructions to check the URL.
- Connection error or 5xx → cached demo fallback from 2023/24 season.
- `openai.AuthenticationError` → "API key rejected" with BYOK recovery steps.
- `openai.RateLimitError` → "Rate limit reached" with retry advice.
- Generic exception → "Something went wrong" as a catch-all.

Each error yields a ChatMessage with a descriptive title instead of letting Gradio show opaque red error pills.

**Rate limiting.** An in-memory counter limits the hosted key to 50 runs per day. It resets on UTC date rollover and on every HF Space restart — no persistence layer, no Redis, no SQLite. This is intentional. The cap prevents sustained abuse within a single uptime window. BYOK runs bypass the cap entirely. The counter increments only after a successful run, so failed attempts (auth errors, API outages) do not consume the budget.

```python
_daily_runs = 0
_day_start: date | None = None
_DAILY_CAP = 50
```

**BYOK threading.** The BYOK key is set via a module-level variable in `llm.py`:

```python
_api_key_override: str | None = None

def set_api_key_override(key: str | None) -> None:
    global _api_key_override
    _api_key_override = key
```

This works because the Gradio handler enforces `concurrency_limit=1`. Only one graph executes at a time, so the global is never contested. The alternative — threading the key through the LangGraph state and modifying every node — would violate the constraint that node files should not change for a UI concern.

**Cached demo fallback.** When the FPL API fails, the generator catches the exception and replays a pre-recorded trace:

```python
except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 404:
        yield _error_yield("Team not found", "...")
    else:
        async for result in _replay_cached_demo():
            yield result
```

The cached demo is the Gyökeres → Beto recovery trace from GW 33 of the 2023/24 season. It replays with 1-second delays between events so it still feels live. A warning banner at the top makes it clear the data is historical.

**Right-column builders are pure functions.** `app_helpers.py` contains five functions that take a state dict and return display-ready strings. No network calls, no LLM calls, no side effects. This makes them trivially testable — construct a dict, call the function, assert on the output.

### 7.4 What confused me

Gradio's `gr.Chatbot` was designed for conversations, not traces. The `metadata` field that controls the pending/done spinner is underdocumented. I discovered it by reading Gradio's source — `status: "pending"` shows a pulsing indicator, `status: "done"` collapses the card. There is no official API for this.

The `gr.skip()` function was also non-obvious. Without it, every yield sends empty strings to the right-column components, which Gradio interprets as "clear and show loading." Using `gr.skip()` tells Gradio "this component has not changed, leave it alone." The difference is invisible in the code but dramatic in the UI.

---

## 8. Phase 6: Evaluation

Phase 6 is not yet implemented. The `eval/` directory exists but contains only an empty `__init__.py`. The `backtest` CLI command is stubbed. This section describes what is planned.

### 8.1 What will be built

`eval/backtest.py` — a harness that runs the agent on historical squad snapshots, one gameweek at a time, with data leakage prevention. `eval/scoring.py` — functions that compare the agent's recommendations against actual outcomes. `eval/heuristic.py` — a deterministic baseline that picks transfers by highest expected points and captains by highest form, with no LLM involved. `eval/judge.py` — an LLM-as-judge that scores the agent's reasoning on factual grounding, logical coherence, and actionability.

### 8.2 Why it will exist

Without evaluation, the project is a demo. With evaluation, it is a system. The backtest answers "does this agent make good decisions?" The heuristic baseline answers "does the LLM add value over a simple rule?" The LLM-judge answers "is the reasoning coherent, or is the agent getting lucky?"

### 8.3 Planned approach

**Backtesting with leakage prevention.** For each gameweek N in a range, the harness fetches the user's squad as of GW N-1 and runs the agent with player history filtered to `round < N`. The agent never sees future data. This is the same constraint that a human manager operates under — you make decisions on Monday with data through Sunday.

**Scoring dimensions.** Transfer delta: did the incoming player outscore the outgoing player in the target gameweek? Captain delta: did the agent's captain outscore the user's actual captain? These are measured per-gameweek and aggregated as hit rates.

**Heuristic baseline.** The deterministic baseline picks the transfer candidate with the highest expected points at the weakest squad position, and the captain with the highest expected points in the post-transfer squad. No LLM, no reasoning, no cost. If the agent cannot beat this baseline, the LLM is not earning its keep.

### 8.4 Known limitations

Historical availability data (`chance_of_playing`, `news`) is point-in-time only. The FPL API does not serve historical values for these fields. Backtests will use current availability, which may not match what was known at the time. This is a data limitation, not a design flaw.

---

## 9. Glossary

**FPL.** Fantasy Premier League — the official fantasy football game run by the Premier League.

**Gameweek (GW).** One round of Premier League fixtures. There are 38 per season.

**FDR.** Fixture Difficulty Rating. A 1–5 score assigned by the FPL team to each fixture, based on team strength and home/away. Does not update dynamically with form.

**Form.** A player's average points per game over the last 30 days. Returned as a string by the FPL API (e.g., "4.7").

**EP Next.** Expected Points Next — the FPL API's prediction of how many points a player will score in the upcoming gameweek.

**now_cost.** A player's current price, stored in tenths. `now_cost = 100` means £10.0m.

**element_type.** Position code from the FPL API: 1 = GKP, 2 = DEF, 3 = MID, 4 = FWD.

**Free transfer (FT).** One free transfer per gameweek, banking up to two. Additional transfers cost 4 points each.

**Replan loop.** The architectural centerpiece: the LLM proposes a transfer, the constraint engine validates it, and if it fails, the LLM is asked to try again with the violation list injected into the prompt. Capped at 2 attempts.

**BYOK.** Bring Your Own Key — users can paste their own OpenAI API key to bypass the hosted key's daily cap.

**Structured output.** LangChain's `with_structured_output()` method, which constrains the LLM to return JSON matching a Pydantic schema. Used on all data-returning nodes.

---

## 10. What I'd Do Differently

**Start with the constraint engine, not the data layer.** Phase 1 built the FPL client and candidate filter. Phase 2 built the constraint engine. In retrospect, building the constraint engine first would have forced clearer thinking about what data it needs, and the data layer could have been shaped to serve it. Instead, I built the data layer speculatively and had to adjust field names later.

**Use a custom fixture difficulty model.** The FPL API's FDR is static and does not reflect recent form. A rolling xG-based difficulty score would produce more accurate recommendations. This is the single highest-impact improvement for recommendation quality, and it does not require changing the architecture — just replacing one number in the candidate filter.

**Build the eval harness earlier.** Without evaluation, every prompt change is guesswork. I tuned prompts by reading output and deciding if it "sounded right." An automated scoring function would have caught regressions and quantified improvements. The eval framework should have been Phase 4, not Phase 6.

**Do not approximate selling price.** Using `now_cost` as selling price causes budget violations that should not happen and misses violations that should. Reconstructing the actual selling price from transfer history is feasible — the API exposes the data — but I scoped it out as an MVP trade-off. For a production system, this would be the first thing to fix.
