# GCA-011 — Record findings during play and write the role's report automatically at the end

| Field | Value |
|-------|-------|
| **Epic** | Reporting |
| **Type** | Feature |
| **Priority** | High |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `agent-loop` |

## Summary
For the mock review the report had to be rebuilt from narration afterwards. The agent should log findings as structured data while playing and produce the report itself.

## Requirements & Scope
- Step output gains `findings`: type (bug / ux / positive / note), severity, title, details, confirmed.
- Each finding is stored with the step number, the recent actions (repro) and the screenshot at that moment (`backend/Reports/<session>/`).
- When the run ends (stop, abort, error), one model call writes the role's report from findings + goal outcomes + stats: QA → bug report, Reviewer → mock review, Gamer → run summary, Speedrunner → route summary. Saved as Markdown and announced on the dashboard log.
- `GET /api/report` returns the latest report path and text.

## Acceptance Criteria
- [x] Findings with screenshots are saved during a live run
- [x] Stopping the agent produces a role-specific report file
- [ ] QA report separates confirmed and unconfirmed bugs

## Work Log
- **2026-10-25**: Filed from the Ski Resort Tycoon review run (https://claude.ai/artifact/G8CDDghY7A5QhLUMEvNJgE) with Björn's feedback; started by Claude Code.
- **2026-10-25**: Built: `backend/reporting.py`, findings in the step schema, report written in `agent_loop` at run end, `GET /api/report`. Both reruns wrote `Reports/<session>/report_reviewer.md` + findings.json + one screenshot per finding. Rerun A logged 83 findings (spam); capped at 2 per step and the recorded titles are now in the prompt — rerun B logged 37 in 33 steps. QA report wording is in place but not run live.
