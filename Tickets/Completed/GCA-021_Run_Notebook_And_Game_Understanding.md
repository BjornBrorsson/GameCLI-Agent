# GCA-021 — Run notebook, click checks, turn awareness, optional second frame

| Field | Value |
|-------|-------|
| **Epic** | Game understanding |
| **Type** | Feature |
| **Priority** | High |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | GCA-019, GCA-020 |
| **Labels** | `agent-loop`, `prompting` |

## Summary
Björn's Heroes 3 HD run (GPT-6 Luna via OpenRouter, 119 steps): aiming was fine (crosshairs and 41 insets confirmed the intended targets), but only 36 steps made progress, a battle was lost, and no memory was saved. The model re-derived two-click movement 8 times, clicked while the wrong unit was active, and repeated actions the game had already refused ("Cannot recruit. You already have a Hero in this town").

## Acceptance Criteria
- [x] Run notebook: `notes_add` in the step output; notes deduped, coordinate-free, capped at 30; shown in every prompt; saved as game-scoped memories at the end of the run (`backend/notebook.py`)
- [x] Game messages that state a rule go into the notebook (prompt rule)
- [x] "Before acting" rule: when unsure, read what the game says a click will do (status line, tooltip, cursor, right-click info) first
- [x] Turn awareness: `actor_now` in the step output; plan only for the active actor
- [x] Optional second frame ~0.4 s later (`dual_frame`, dashboard checkbox "Send two frames per step"), for all providers; prompt explains it
- [x] Unit tests (48 passing)
- [ ] Next Heroes 3 run: compare progress rate and repeated discoveries

## Work Log
- **2026-10-26**: Built by Claude Code. Live check on Ski Resort Tycoon with two frames on: the first step produced a note and ran fine through OpenRouter with two images (7.7 s), then the run stopped because a separately launched installed build of the game ("Ski Resort Tycoon", not the debug window) covered the target — the input guard blocked the click as intended.
