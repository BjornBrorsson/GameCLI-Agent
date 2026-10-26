# GCA-008 — Fix `run_macro` NameError, unused revalidate placeholder, and verifier coordinate jumps

| Field | Value |
|-------|-------|
| **Epic** | Reliable actions |
| **Type** | Bug |
| **Priority** | Low |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | S |
| **Depends on** | — |
| **Labels** | `agent-loop`, `bug` |

## Summary
Small defects found while reading the loop:
1. `run_macro` passes an undefined `scale` (`agent_loop.py:445, 447`), so any macro raises NameError (caught by the outer loop as a generic error).
2. `revalidate_actions` substitutes `{commands_list_short}`, which `REVALIDATE_PROMPT` doesn't contain (`llm.py:862`).
3. Template matching "corrects" good coordinates by large jumps at low confidence: a correct close-button click moved 45 px up, a hover moved 120 px at 0.36 confidence (`action_verifier.verify_and_adjust`).

## Requirements & Scope
- Use `scale_x, scale_y` in the macro path.
- Add the placeholder to the prompt or remove the substitution.
- Only apply a verifier shift when confidence ≥ threshold and the shift is under ~20 px; log and keep the original otherwise.

## Acceptance Criteria
- [ ] A recorded macro replays without error
- [x] Revalidate prompt lists the available commands
- [x] Verifier never moves a target more than the configured limit

## Work Log
- **2026-10-25**: Filed by Claude Code from the Ski Resort Tycoon playtest (https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ).
- **2026-10-25**: Built (Claude Code). `run_macro` now passes scale_x/scale_y; REVALIDATE_PROMPT has `{commands_list_short}`; verifier corrections over MAX_SHIFT_PX (24 px) are rejected and logged (2 rejected in the live run). Also fixed: COMBINED_AGENT_PROMPT showed literal `{{ }}` braces; CLI output decoded as UTF-8 (tooltips were mojibake like 'â€”'); auth/billing errors (401/402/403) now stop the run instead of retrying forever. Macro replay not exercised (no macros recorded).
