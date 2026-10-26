# GCA-009 — Change approach when stuck: escalate to a stronger model, then move on

| Field | Value |
|-------|-------|
| **Epic** | Reliable actions |
| **Type** | Feature |
| **Priority** | High |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `agent-loop` |

## Summary
In the Ski Resort Tycoon review run 49 of 79 steps made no progress, mostly repeating piste attempts with 43 Escape presses. After 3 no-progress steps the agent only paused; resuming changed nothing.

## Requirements & Scope
- After `NO_PROGRESS_ESCALATE` (3) no-progress steps, plan the next step with a stronger model from the same provider (Sonnet tier), with a "you are stuck — summary of the attempts — choose a different approach" block.
- If the goal still makes no progress after the escalation steps, and the role allows skipping (QA Tester, Reviewer, Explorer), mark the current goal skipped, record a finding, and continue with the next goal (GCA-010).
- Gamer / Speedrunner never skip; they keep escalating and pause only after a long no-progress streak.
- Game agnostic: vision, reasoning and computer use only — no game-specific code, game state hooks or image preprocessing (Björn, 2026-10-25).

## Acceptance Criteria
- [x] Escalation model is used after 3 no-progress steps (visible in the log)
- [x] Reviewer/QA runs skip a goal after the escalation fails and continue
- [x] Gamer runs never skip a goal
- [x] Unit tests cover the escalation/skip decisions

## Work Log
- **2026-10-25**: Filed from the Ski Resort Tycoon review run (https://claude.ai/artifact/G8CDDghY7A5QhLUMEvNJgE) with Björn's feedback; started by Claude Code.
- **2026-10-25**: Built: `backend/goals.py` (stuck policy) + escalation in `agent_loop.py`; stronger model per provider in `llm.escalation_model`. Review reruns on Ski Resort Tycoon (OpenRouter, Haiku 5.5, Reviewer): rerun A 36 steps / $0.15, rerun B 33 steps / $0.08. Baseline: first review run, 79 steps / 30 min / $0.15. Rerun A escalated 5 times and skipped 'Build a lift' after 2 Sonnet steps; the agent kept working instead of looping (baseline: 49/79 no-progress steps). Sonnet 5.5 on OpenRouter twice answered with tool-call XML — fixed with structured output + one retry (0 parse errors in rerun B).
