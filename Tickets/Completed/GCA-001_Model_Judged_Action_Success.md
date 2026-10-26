# GCA-001 — Judge action success with the model, not a pixel-diff threshold

| Field | Value |
|-------|-------|
| **Epic** | Reliable actions |
| **Type** | Architecture |
| **Priority** | Critical |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `agent-loop`, `verification` |

## Summary
An action only counts as successful when more than 1.2% of the pixels change (`backend/agent_loop.py:557-560`). Hovers, tooltips, button highlights and speed changes rarely do, so they are marked failed and retried. In the Ski Resort Tycoon playtest 17 of 54 steps had a "failed" action, and in 12 of those the retry helper itself said the action had worked or was not needed. Each false failure costs 5 retries plus an LLM call, and then pauses the agent.

## Requirements & Scope
- Execute the planned batch, then take one fresh screenshot. Give the next planning call the previous plan with each action's *expected* result and ask the model to report which actions did what was expected (`"last_step_check": [...]` in the JSON).
- Remove the per-action retry loop for click/hover/scroll. Keep the pixel diff only as a hint ("screen changed: yes/no") in the prompt.
- Keep a single cheap retry only for drags (snap-back detection is still useful).
- Treat `retry_assist` → `skip` as success when its reason says the goal is already met (`agent_loop.py:625-628`).
- Pause only after N consecutive steps (default 3) where the model reports no progress, not on any single failure (`agent_loop.py:829-835`).
- Drop the fixed-offset `ActionVerifier.nudge_action` retries; on the speed bar they clicked neighbouring buttons.

## Acceptance Criteria
- [x] Hover-only and highlight-only actions are no longer logged as FAILED
- [x] The struggle pause triggers on lack of progress over several steps, not on one failed action
- [x] Re-run the Ski Resort Tycoon playtest instructions: fewer than 3 pauses in 20 steps
- [x] Narration log shows the model's check of the previous step

## Work Log
- **2026-10-25**: Filed by Claude Code from the Ski Resort Tycoon playtest (https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ).
- **2026-10-25**: Built (Claude Code). Per-action retry loop, nudges, retry-assist, revalidate and verify-agent calls removed from the loop; each action runs once and its expectation + harness outcome go into the next prompt as PREVIOUS STEP. The model returns `last_step_check` and `progress`; the agent pauses only after 3 no-progress steps or a blocked input (`agent_loop.py`, `llm.py` STEP_CHECK_RULES). Drags keep one snap-back retry.
- **2026-10-25**: Live run on Ski Resort Tycoon (Claude Code CLI + Haiku, 40 steps, 14 min): 0 actions falsely logged as failed, 3 no-progress pauses, first ever lift built by the agent. Before: 15 pauses and 17 false failures in 54 steps.
