# GCA-015 — Mark where the previous step's clicks landed on the next screenshot

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
The model couldn't tell how far off a missed click was. Numbered red crosshairs now show where each mouse action of the previous step landed, matching the numbers in PREVIOUS STEP.

## Acceptance Criteria
- [x] Crosshairs drawn on the model's copy only (template matching uses the clean frame)
- [x] Prompt explains them and asks to correct by the offset
- [x] Unit test

## Work Log
- **2026-10-25**: Filed and built by Claude Code after Björn asked to focus on computer use; game agnostic (vision, reasoning and computer use only).
- **2026-10-25**: Built: `ScreenCapture.draw_click_markers` + `click_points` in `agent_loop`. In the dashboard smoke test the model referred to the marker directly ("the chairlift button is highlighted with the red crosshair '1' on it").
