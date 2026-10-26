# GCA-012 — Role-dependent verification: QA reproduces bugs, other roles report experience

| Field | Value |
|-------|-------|
| **Epic** | Reporting |
| **Type** | Feature |
| **Priority** | Normal |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `agent-loop` |

## Summary
Several complaints in the mock review may have been the agent's own input mistakes. How much effort goes into separating agent error from game bugs depends on the role (Björn, 2026-10-25).

## Requirements & Scope
- QA Tester: before a bug is `confirmed`, reproduce it once with a different input method (key vs click, centre of the element); unconfirmed bugs are labelled so in the report.
- Reviewer / Gamer / Speedrunner: no reproduction steps; findings say when the problem may be the agent's own input.
- Implemented in the role prompts and the report writer.

## Acceptance Criteria
- [x] Tester prompt asks for reproduction before confirming
- [x] Reviewer prompt asks to flag possible own mistakes instead of reproducing
- [x] Report labels confirmed vs unconfirmed

## Work Log
- **2026-10-25**: Filed from the Ski Resort Tycoon review run (https://claude.ai/artifact/G8CDDghY7A5QhLUMEvNJgE) with Björn's feedback; started by Claude Code.
- **2026-10-25**: Built in `ROLE_PROMPTS` and the report prompts. Reviewer reports now say where a problem may be the agent's own input (seen in both reports). QA mode not run live yet.
