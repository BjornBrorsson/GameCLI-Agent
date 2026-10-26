# GCA-007 — Wait for the screen to settle before capturing a new step

| Field | Value |
|-------|-------|
| **Epic** | Reliable actions |
| **Type** | Bug |
| **Priority** | Normal |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | S |
| **Depends on** | — |
| **Labels** | `agent-loop` |

## Summary
After clicking SANDBOX the next step captured the screen mid-load, planned for the title screen, and then clicked into the loaded world with stale coordinates. The loop captures immediately after a fixed 1.5 s cooldown (`agent_loop.py:231`).

## Requirements & Scope
- Before each step's capture, call `_wait_for_stable` (pHash tolerant of idle animation) with a short timeout, e.g. 3 s.
- If the screen is still changing at timeout, capture anyway and tell the model "screen was still changing".

## Acceptance Criteria
- [ ] After a scene change the planning screenshot shows the new scene
- [x] No added delay when the screen is already stable

## Work Log
- **2026-10-25**: Filed by Claude Code from the Ski Resort Tycoon playtest (https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ).
- **2026-10-25**: Built (Claude Code). Each step waits up to 3 s for a stable pHash before the planning screenshot and tells the model when the screen was still changing. The fixed 1.5 s cooldown is gone, so stable screens cost ~0.8 s. Scene-change case (title → sandbox) not re-run live, since this run started in Sandbox.
