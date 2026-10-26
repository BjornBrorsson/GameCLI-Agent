# GCA-010 — Goal checklist with step budgets, derived from the user's instructions

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
A review should cover the whole game, not 30 minutes on one tool. The agent should split the user's instructions (whatever the game) into ordered goals and track them.

## Requirements & Scope
- At run start, one model call turns the instructions into 3–8 ordered goals (game agnostic, no per-game lists).
- Each step's prompt shows the checklist (done / current with steps used / pending / skipped); the model reports `goal_done`.
- Default budget per goal (e.g. 15 steps) for exploration roles; exceeding it triggers GCA-009.
- Checklist outcome goes into the end-of-run report (GCA-011).
- Game agnostic: vision, reasoning and computer use only — no game-specific code, game state hooks or image preprocessing (Björn, 2026-10-25).

## Acceptance Criteria
- [ ] Goals are generated from any instruction text (tested with two different games' instructions)
- [x] Current goal and budget appear in each prompt
- [x] Goal outcomes appear in the report

## Work Log
- **2026-10-25**: Filed from the Ski Resort Tycoon review run (https://claude.ai/artifact/G8CDDghY7A5QhLUMEvNJgE) with Björn's feedback; started by Claude Code.
- **2026-10-25**: Built: `GoalTracker` + `llm.plan_goals` (goals come from the instructions and first screenshot only). Rerun B finished all goals and ended the run itself. Two issues seen: the planner once merged 'paths, a lift, a piste and a building' into one goal that ran out of budget (prompt now asks for one thing per goal), and a goal skipped earlier can't be marked done when it is completed later (follow-up). Tested on Ski Resort Tycoon only; 'with two different games' instructions' still open.
