# GCA-020 — Auto-zoom inset after a missed click

| Field | Value |
|-------|-------|
| **Epic** | Computer use |
| **Type** | Feature |
| **Priority** | High |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | S |
| **Depends on** | GCA-014, GCA-015 |
| **Labels** | `computer-use` |

## Summary
In the GFN Olden Era run, hero-move clicks missed the path marker by ~26 image px (~80 screen px at 3x) and the model only zoomed once on its own. Now, when a click changes nothing on screen, the next screenshot gets a red-framed picture-in-picture inset: a 3x enlargement around that click, with its crosshair and rulers in full-screen coordinates, placed in the corner farthest from the click. No extra model call, game agnostic.

## Acceptance Criteria
- [x] Inset added automatically after a click with no visible change (`ScreenCapture.add_miss_inset`)
- [x] Inset rulers in full-screen coordinates; prompt note explains it
- [x] Unattended (timed) runs don't pause on long no-progress streaks — they keep trying until the limit
- [x] Model goal revisions keep the replaced goal in the history
- [ ] Measure on the next GFN run whether hero-move clicks land more often

## Work Log
- **2026-10-26**: Built by Claude Code. 8-minute Ski Resort Tycoon check: 2 insets shown, both after clicks the game rejected for rule reasons (uphill lift leg, blocked piste), which the model then read correctly from the inset area. Lift built; piste still not.
