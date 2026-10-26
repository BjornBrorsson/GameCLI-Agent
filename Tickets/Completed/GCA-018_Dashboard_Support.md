# GCA-018 — Dashboard: capture mode, goals, findings, checks and the end-of-run report

| Field | Value |
|-------|-------|
| **Epic** | Reporting |
| **Type** | Feature |
| **Priority** | High |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `computer-use` |

## Summary
The dashboard hid the new information (goal progress, stuck handling, findings, step checks) and had no way to open the report.

## Acceptance Criteria
- [x] Screen Capture selector sent as `capture_mode`
- [x] GOAL / CHECK / FINDING / REPORT log lines with their own tags
- [x] 'View latest report' button and report modal (also opens when a REPORT line arrives)
- [x] Every log line appeared twice (two WebSockets under React StrictMode) — fixed

## Work Log
- **2026-10-25**: Filed and built by Claude Code after Björn asked to focus on computer use; game agnostic (vision, reasoning and computer use only).
- **2026-10-25**: Built in `frontend/src/App.jsx` + `index.css`; `vite build` passes. Smoke-tested in a browser: started a Reviewer run from the dashboard (Claude Code CLI), saw goals, checks, findings, stopped it, report written and viewable.
