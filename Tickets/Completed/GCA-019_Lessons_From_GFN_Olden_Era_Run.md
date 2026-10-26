# GCA-019 — Fixes from the GeForce NOW Olden Era run

| Field | Value |
|-------|-------|
| **Epic** | Reliable actions |
| **Type** | Bug |
| **Priority** | High |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | S |
| **Depends on** | GCA-009, GCA-010, GCA-017 |
| **Labels** | `agent-loop`, `goals`, `safety` |

## Summary
Björn's 30-minute GFN run (Heroes of Might and Magic: Olden Era, Gamer role) exposed four problems:
1. The goal planner saw only a loading screen and invented RTS goals (Build a Farm / Barracks / destroy the Castle). 65 steps went to an impossible Farm goal; the scenario's real objective ("Reach the Cat Paradise") was never adopted.
2. The stronger model was used on 39 consecutive stuck steps, which made up most of the $1.04 cost.
3. A stray click hit "Exit to Main Menu"; only the game's confirmation dialog saved the session. "exit" / "main menu" were not high-risk keywords.
4. Stopping a run while it was paused left the dashboard showing "paused".

## Acceptance Criteria
- [x] The model can replace the remaining goals (`goals_update`) when they don't fit the game it sees; done/skipped goals stay as history
- [x] The stronger model plans at most `ESCALATION_STEPS` steps per stuck streak
- [x] "exit", "main menu", "surrender", "concede", "resign", "retire" are high-risk; in timed (unattended) runs such clicks are refused and the model is told to pick a safer option instead of waiting for an approval nobody gives
- [x] `stop()` clears the paused state
- [x] Unit tests (42 passing)
- [ ] Re-run on GFN to confirm the goal revision kicks in

## Work Log
- **2026-10-26**: Built by Claude Code from the run's report and log (`Reports/20261026_082614`, `Logs/Console_20261026_082614.log`).
