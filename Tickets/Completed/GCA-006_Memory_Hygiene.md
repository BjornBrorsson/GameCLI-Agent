# GCA-006 — Memory hygiene: game-scoped facts, no pixel coordinates, cap and dedupe

| Field | Value |
|-------|-------|
| **Epic** | Memory |
| **Type** | Feature |
| **Priority** | Normal |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | S |
| **Depends on** | — |
| **Labels** | `memory` |

## Summary
After the playtest `backend/memories.json` holds 18 memories, all saved with `game=""` and all injected into every prompt (`memories.py format_for_prompt`). Many contradict each other ("Clear All didn't work" then "Clear All works"; three different X positions for one window), and pixel coordinates go stale as soon as a window moves.

## Requirements & Scope
- Save memories with the game id (window title or a game field from the start request).
- Prompt the model to store facts and hotkeys ("Path tool is hotkey 1"), not coordinates; reject memories that are mostly coordinates.
- When a new memory contradicts an older one on the same subject, replace it.
- Inject at most the 10 most relevant memories (by game + recency + similarity).
- Dashboard: show and delete memories per game (endpoints exist in `main.py`).

## Acceptance Criteria
- [x] Memories are tagged by game and filtered by it
- [x] Prompt includes ≤ 10 memories
- [x] A contradicting memory replaces the old one in a test

## Work Log
- **2026-10-25**: Filed by Claude Code from the Ski Resort Tycoon playtest (https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ).
- **2026-10-25**: Built (Claude Code). `MemoryStore.save_from_agent`: game-tagged (window title minus build tags), refuses text with pixel coordinates, `save_memory replaces=<id> …` corrects an entry, near-duplicates (ratio ≥ 0.85) replace. Prompt shows ids, newest 10, skips old coordinate memories; untagged agent memories only show for the game they name. Tests in `backend/tests/test_playtest_fixes.py`. Live run: 11 added, 1 rejected for coordinates.
