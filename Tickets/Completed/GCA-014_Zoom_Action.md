# GCA-014 — Zoom action: enlarge a small screen region before aiming

| Field | Value |
|-------|-------|
| **Epic** | Computer use |
| **Type** | Feature |
| **Priority** | High |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `computer-use` |

## Summary
Clicks on small or crowded targets missed (map nodes in Slay the Spire 2, piste points in Ski Resort Tycoon). The model can now ask to see a region enlarged; the zoomed view's rulers are labelled in full-screen coordinates, so commands stay in one coordinate space.

## Acceptance Criteria
- [x] `zoom x1 y1 x2 y2` command; next screenshot is the enlarged region (native resolution, aspect kept)
- [x] Rulers on the zoomed view show full-screen coordinates
- [x] Boxes wider than 480 px are refused (the model asked for full-screen 'zooms' twice)
- [x] Unit tests

## Work Log
- **2026-10-25**: Filed and built by Claude Code after Björn asked to focus on computer use; game agnostic (vision, reasoning and computer use only).
- **2026-10-25**: Built: `ScreenCapture.zoom_view`, ruler origin/scale in `draw_rulers`, handling in `agent_loop`. Live run on Ski Resort Tycoon (OpenRouter Haiku 5.5, Reviewer, 42 steps, 10.5 min, $0.19): built a path, a lift and a building and finished 5 of 7 goals; 62% of steps made progress (first review run: 38%, 43 Escape presses vs 9 now). The model used zoom 4 times; 2 were useless full-screen boxes, now refused.
