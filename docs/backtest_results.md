# Backtest Results — Phase 6 Evaluation

## TL;DR

- **Transfers: heuristic wins.** The agent averaged -1.0 pts/GW on transfers versus the heuristic's +1.7 (hit rates 29% vs 43%). The agent over-trades — 81% transfer rate — and routinely sells established players for speculative picks that blank. This gap is robust: the agent's transfer delta is negative even after removing outlier weeks.
- **Captaincy: agent slightly ahead, but fragile.** Agent captain delta -0.4 vs heuristic -1.3, but the entire advantage comes from two outlier GWs (17 and 24, combined +23). Remove them and the agent drops to -1.6. On a different 21-GW window this comparison could plausibly invert.
- **Most actionable finding: the LLM-judge needs cross-family evaluation.** All 21 coherence scores fell in [4.0, 5.0] with sigma = 0.33. The GPT-4o judge scored the agent's worst decisions (transfer deltas of -11 and -8) at 5.0. Same-model-family evaluation is not discriminating — the judge infrastructure works, but the scores are not trustworthy until re-run with a cross-lineage model.

**Sections:** [Setup](#setup) · [Per-GW Results](#per-gw-results) · [Summary](#summary) · [Coherence Distribution](#coherence-score-distribution) · [Analysis](#analysis) · [Failure Modes](#failure-modes-observed)

## Setup

- **Team**: 44
- **Gameweek range**: GW 5–25 (21 gameweeks)
- **Agent LLM**: GPT-4o-mini (via `openai` provider)
- **Judge LLM**: GPT-4o (fallback — `ANTHROPIC_API_KEY` was not set, so the judge used the same model family as the agent instead of the intended cross-family Claude Sonnet)
- **Heuristic baseline**: Form x (6 - fixture difficulty) composite; same-position transfer, post-transfer team-limit check
- **Date run**: 2026-04-15

### Limitations

- `ep_next` is approximated as computed 5-GW form average (FPL's model is proprietary).
- Player availability (`status`, `news`) is not historically reconstructable. All players are set to `status="a"` during backtests, so availability-based constraint violations (Rule 7) never fire. This means the replan loop triggers less often than it would in a live run.
- Selling price is approximated as `now_cost` (same as the live MVP).
- The judge used GPT-4o instead of Claude Sonnet due to a missing API key. Same-model-family evaluation may inflate coherence scores (no cross-family bias correction). This is a known gap in this run.

## Per-GW Results

| GW | Agent Action | Agent Out | Agent In | Transfer Delta | Agent Captain | Cap Delta | H Out | H In | H Transfer Delta | H Captain | H Cap Delta | Coherence | Replans |
|----|-------------|-----------|----------|---------------|--------------|-----------|-------|------|-----------------|-----------|-------------|-----------|---------|
| 5 | transfer | Mateta | Woltemade | **-4** | M.Salah | 0 | Semenyo | Anthony | +5 | Lacroix | -3 | 4.3 | 1 |
| 6 | transfer | Mateta | Joao Pedro | 0 | M.Salah | 0 | Andersen | Guehi | +4 | Semenyo | +5 | 4.7 | 1 |
| 7 | transfer | Raya | Roefs | **-4** | Haaland | 0 | L.Paqueta | Enzo | +1 | Haaland | 0 | 4.3 | 0 |
| 8 | hold | -- | -- | 0 | Haaland | 0 | Dewsbury-Hall | Xhaka | +3 | Haaland | 0 | 4.7 | 2 |
| 9 | transfer | Senesi | Mukiele | **-3** | Haaland | 0 | Lacroix | Mukiele | -2 | Gabriel | +7 | 4.7 | 1 |
| 10 | transfer | B.Fernandes | Mbeumo | **-3** | Haaland | 0 | Evanilson | Welbeck | +5 | Gabriel | -1 | 4.7 | 0 |
| 11 | hold | -- | -- | 0 | Haaland | 0 | Virgil | James | 0 | Gabriel | -3 | 5.0 | 2 |
| 12 | transfer | Mateta | Joao Pedro | 0 | Haaland | 0 | hold | -- | 0 | Gabriel | -2 | 4.0 | 2 |
| 13 | transfer | Raya | Martinez | **+4** | Haaland | -3 | hold | -- | 0 | Haaland | -3 | 4.7 | 0 |
| 14 | transfer | Collins | Lacroix | **+7** | Haaland | 0 | Collins | Lacroix | +7 | Lacroix | -6 | 5.0 | 0 |
| 15 | hold | -- | -- | 0 | Haaland | **-16** | hold | -- | 0 | Haaland | -16 | 4.7 | 0 |
| 16 | transfer | Dewsbury-Hall | Kamara | 0 | Saka | -2 | Marc Guiu | Nmecha | 0 | Saka | -2 | 4.0 | 1 |
| 17 | transfer | Cunha | Wilson | **-7** | Haaland | **+13** | hold | -- | 0 | Foden | 0 | 5.0 | 0 |
| 18 | hold | -- | -- | 0 | Haaland | 0 | hold | -- | 0 | Foden | 0 | 4.3 | 0 |
| 19 | transfer | L.Paqueta | Wilson | **-11** | Haaland | 0 | hold | -- | 0 | Haaland | 0 | 5.0 | 2 |
| 20 | transfer | Gabriel | Matheus N. | **-8** | Haaland | 0 | hold | -- | 0 | Haaland | 0 | 5.0 | 0 |
| 21 | transfer | Hall | Collins | **+9** | Haaland | 0 | hold | -- | 0 | Cunha | -4 | 5.0 | 0 |
| 22 | transfer | Szoboszlai | Janelt | +1 | Haaland | 0 | hold | -- | 0 | Thiago | 0 | 5.0 | 1 |
| 23 | transfer | Tavernier | Schade | +2 | Haaland | 0 | Marc Guiu | Barry | +9 | Thiago | +1 | 5.0 | 0 |
| 24 | transfer | Raya | Pickford | **-4** | B.Fernandes | **+10** | Marc Guiu | Brobbey | +2 | Enzo | +8 | 5.0 | 0 |
| 25 | transfer | Gabriel | Saliba | +1 | Saka | **-10** | Marc Guiu | Barry | +2 | Enzo | -8 | 4.3 | 0 |

## Summary

| Metric | Agent (mean +/- sigma) | Heuristic (mean +/- sigma) |
|--------|------------------------|---------------------------|
| Transfer delta | -1.0 +/- 4.5 | +1.7 +/- 2.7 |
| Transfer hit rate | 29% | 43% |
| Captain delta | -0.4 +/- 5.5 | -1.3 +/- 4.9 |
| Captain hit rate | 10% | 19% |
| Coherence (1-5) | 4.68 +/- 0.33 | -- |
| Hold rate | 19% | -- |
| Avg replans | 0.6 | -- |
| First-attempt valid rate | 57% | -- |
| GWs evaluated | 21 | -- |

## Coherence Score Distribution

All 21 coherence scores fell between 4.0 and 5.0. The distribution:

| Score | Count | % |
|-------|-------|---|
| 5.0 | 9 | 43% |
| 4.67 | 6 | 29% |
| 4.33 | 4 | 19% |
| 4.0 | 2 | 10% |

- **Range**: 4.0 – 5.0
- **Median**: 4.67
- **Mean**: 4.68

No scores fell below 4.0. No judge explanation flagged a factual contradiction or logical incoherence.

**Important caveat**: The judge used GPT-4o (same family as the GPT-4o-mini agent) because `ANTHROPIC_API_KEY` was not set for this run. Same-model-family evaluation is known to inflate coherence scores — GPT models tend to rate GPT-generated reasoning more favorably than cross-family judges would. The uniformly high scores (nothing below 4.0) may partly reflect this bias rather than genuinely flawless reasoning. A re-run with Claude Sonnet as the judge is needed to validate these scores. **No pass/fail threshold is set.** The narrow distribution (0.33 sigma over a 1–5 scale) suggests the judge is not discriminating well between stronger and weaker reasoning, which further supports the same-family bias hypothesis.

A more rigorous evaluation would use a cross-lineage judge model — Claude Sonnet (the intended default), Llama-405B, or Mistral-Large via their respective APIs — or human evaluation on a sampled subset of gameweeks. This is deferred for cost and time; it represents the highest-priority next step for a v2 evaluation.

## Analysis

### Transfers: heuristic wins clearly

The agent averaged -1.0 points/GW on transfer decisions versus the heuristic's +1.7 — a gap of 2.7 points per gameweek. The agent's transfer hit rate (29%) also lags the heuristic (43%). The agent made transfers in 17 of 21 GWs (81% transfer rate) compared to a lower effective transfer rate from the heuristic, which held more often. The agent appears to over-trade: it acts on reasoning that reads well but often picks the wrong side of a close call, particularly when selling established players for speculative alternatives.

The worst agent transfer decisions — GW 19 (L.Paqueta to Wilson, -11), GW 20 (Gabriel to Matheus N., -8), and GW 17 (Cunha to Wilson, -7) — share a pattern: the agent sold players who went on to return strongly that week, replacing them with lower-owned differentials who blanked. The agent targeted Wilson twice (GW 17 and 19, combined delta -18), suggesting a form-chasing bias toward specific players that the heuristic's simpler composite avoids. In contrast, the heuristic's composite score more often found genuine value: GW 23 (Marc Guiu to Barry, +9) and GW 5 (Semenyo to Anthony, +5) were cases where sorting by form x fixture difficulty identified the right swap.

### Captaincy: agent slightly better, but driven by two outlier GWs

The agent's captain delta (-0.4) was less negative than the heuristic's (-1.3), but this advantage is fragile. It is almost entirely explained by GW 17 (Haaland captain, +13 delta when the user picked Saka who scored 3) and GW 24 (B.Fernandes captain, +10 delta when the user picked Saka who blanked). Remove those two GWs and the agent's captain delta drops to roughly -1.6. The implication is that the agent's captain advantage is not statistically robust; on a different 21-GW window the comparison could plausibly invert.

Both agent and heuristic were crushed by GW 15 (-16 each, when B.Fernandes scored 18 as user captain while both systems picked Haaland who scored 2). The one other time the agent deviated from Haaland — GW 25, captaining Saka who blanked (delta -10) — the decision backfired. The agent's tendency to default to Haaland as captain (17 of 21 GWs) is safe but rarely differentiating — it matches the user's pick most of the time, producing captain delta = 0.

### Replan loop is active and functioning

The replan loop fired in 9 of 21 GWs (43%). Three GWs exhausted all 2 replan attempts (GW 8, 11, 19), after which the system correctly cleared the proposal and proceeded to captain selection. First-attempt validity was 57%, meaning 43% of initial LLM proposals violated at least one constraint. This confirms the architectural decision to keep the candidate filter loose and delegate enforcement to the validator — the replan loop is not a dead feature, it is genuinely load-bearing.

### What this means for production

The replan loop architecture, constraint engine, and evaluation harness are production-ready — they work correctly, surface real violations, and provide measurable feedback. The transfer recommendation quality is not: the agent underperforms a simple heuristic, and the current LLM-judge cannot reliably distinguish good reasoning from bad. A v2 would address both by (a) adding a "hold unless confident" prior to reduce the over-trading bias, and (b) deploying the cross-family judge to get trustworthy coherence signal that can inform prompt tuning.

## Failure Modes Observed

- **Agent over-trades into negative-EV swaps (GW 19: -11, GW 17: -7).** The agent's worst transfers share a pattern: selling an established player for a speculative differential who blanks. GW 19 (L.Paqueta to Wilson, -11) is the most extreme — L.Paqueta scored 13 that week while Wilson scored 2. The agent needed 2 replans to produce a valid proposal, suggesting the initial candidates were poor. The same Wilson was targeted in GW 17 (Cunha to Wilson, -7, Cunha scored 9), revealing a form-chasing bias toward specific players across multiple weeks. Despite the poor outcomes, the judge scored both GWs at 5.0 — highlighting the gap between "well-reasoned" and "correct." Coherent reasoning grounded in historical form cannot predict single-GW explosions.

- **Agent sells premium defenders who haul (GW 20: -8).** Gabriel scored 10 points that week; Matheus N. scored 2. The agent sold a premium Arsenal defender for a speculative pick, while the heuristic held (delta 0) and avoided the loss entirely. This illustrates the agent's eagerness to act — 81% transfer rate versus the heuristic's lower rate. Sometimes the best move is no move, and the agent lacks a "hold unless confident" prior that would prevent negative-EV swaps.

- **Captain defaults mask a thin advantage (GW 15: -16, GW 25: -10).** Both agent and heuristic picked Haaland in GW 15 when the user had B.Fernandes, who scored 18 (Haaland scored 2) — a variance event neither system could predict, but it dominates the captain averages. In GW 25, the agent deviated from Haaland to captain Saka, who blanked (0 pts) while the user's B.Fernandes scored 10 (delta -10). The agent's two captain wins (GW 17: +13, GW 24: +10) are real but represent outlier variance of the same magnitude. Remove any two of the four extreme captain GWs and the comparison flips direction.

- **Same-family judge bias renders coherence scores uninformative.** All 21 coherence scores fell in [4.0, 5.0] with sigma = 0.33. The GPT-4o judge scored the agent's worst transfer decisions (GW 19: -11, GW 20: -8) at 5.0, indicating it rewards well-structured prose regardless of decision quality. The near-ceiling distribution with minimal variance means the judge is not meaningfully discriminating between strong and weak reasoning. A more rigorous evaluation would use a cross-lineage judge model — Claude Sonnet (the intended default), Llama-405B, or Mistral-Large — or human evaluation on a sampled subset. This is the most actionable finding: the judge infrastructure works, but the scores are not trustworthy until re-run with a cross-family model.
