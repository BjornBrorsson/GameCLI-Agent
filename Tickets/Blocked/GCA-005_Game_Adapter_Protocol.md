# GCA-005 — Game adapter protocol: click UI elements by name and read game state

| Field | Value |
|-------|-------|
| **Epic** | Game adapters |
| **Type** | Architecture |
| **Priority** | High |
| **Status** | Blocked |
| **Assignee** | — |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `adapter`, `protocol` |

## Summary
Most failures in the playtest came from guessing pixels and guessing whether an action worked. When a game can expose its UI and state (debug builds, mods, our own games), the agent should use it. Ski Resort Tycoon ticket SKTY-157 proposes the game side for Godot: `/ui` (visible Controls with path, text, tooltip, rect, state) and `/state` (speed, paused, funds, tool, open windows, last toast).

## Requirements & Scope
- Define the protocol (JSON over localhost HTTP) in `docs/` and share it with SKTY-157.
- Agent side: optional adapter URL per game; when present, include a compact element list in the prompt and allow `click_element <path>`; verify success from `/state` diffs.
- Falls back to vision when no adapter is configured.
- Optional launcher hook per game (e.g. Ski Resort Tycoon `-- --auto SCN-01 sandbox --qa-port 7777`).

## Acceptance Criteria
- [ ] Protocol doc reviewed by Björn
- [ ] Agent can run the Ski Resort Tycoon playtest through the adapter once SKTY-157 lands
- [ ] No regression in vision-only mode

## Work Log
- **2026-10-25**: Filed by Claude Code from the Ski Resort Tycoon playtest (https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ).
- **2026-10-25**: Won't do as written — Game agnostic: vision, reasoning and computer use only — no game-specific code, game state hooks or image preprocessing (Björn, 2026-10-25). Reading game state through an adapter breaks that rule. Kept here only as a record.
