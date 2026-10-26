# GCA-013 — Cheaper steps: one model call, fewer hovers, waits and per-action overhead

| Field | Value |
|-------|-------|
| **Epic** | Speed |
| **Type** | Performance |
| **Priority** | High |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `agent-loop` |

## Summary
Review run: ~15 s per step with two OpenRouter calls (Lead + Worker), 57 hovers before clicks and 47 model-requested waits. Slay the Spire 2 run: 20 s of execution overhead per step (OCR risk check, several full-resolution captures and settle waits per action).

## Requirements & Scope
- Use the single combined planning call for API providers too (Lead/Worker kept behind an option).
- Prompt: hover only to read a tooltip, not before every click; avoid waits (the harness already settles); cap model waits at 2 s.
- OCR risk check only for clicks; reuse the step's phase detection; cheaper resampling for diff-only captures; shorter post-action settle.

## Acceptance Criteria
- [x] API provider makes one call per step by default
- [x] Execution overhead per mouse action measured before/after
- [x] Median step time on the review instructions measured before/after

## Work Log
- **2026-10-25**: Filed from the Ski Resort Tycoon review run (https://claude.ai/artifact/G8CDDghY7A5QhLUMEvNJgE) with Björn's feedback; started by Claude Code.
- **2026-10-25**: Built: combined single call for API providers (Lead/Worker behind `two_call`), wait cap 2 s, prompt rules against hover-before-click, OCR risk check only on clicks, phase OCR once per step, bilinear resampling for diff captures, shorter post-action settle. Measured per step (settle+capture / model / execute): Slay the Spire 2 run 3.9 / 25.0 / 20.3 s; review rerun B 2.7 / 8.1 / 5.5 s. The first review run averaged ~15 s of model time per step with two calls.
