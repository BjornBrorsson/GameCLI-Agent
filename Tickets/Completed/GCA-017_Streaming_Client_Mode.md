# GCA-017 — Streaming mode for GeForce NOW and other cloud clients

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
Cloud-streamed games are video: PrintWindow can return black or stale frames, input arrives later, and keys posted to the client window never reach the game.

## Acceptance Criteria
- [x] Auto-detects streaming clients by process / window title (GeForce NOW, Xbox, Shadow, Boosteroid, Parsec, Moonlight)
- [x] Uses screen capture, longer hover/hold, longer settle after clicks; no PostMessage key fallback
- [x] Dashboard 'Screen Capture' option (Auto / Screen) to force screen capture
- [x] Verified on GeForce NOW (Heroes of Might and Magic: Olden Era)

## Work Log
- **2026-10-25**: Filed and built by Claude Code after Björn asked to focus on computer use; game agnostic (vision, reasoning and computer use only).
- **2026-10-25**: Built in `screen_capture.py` (`is_streaming_client`, capture_mode), `input_controller.set_streaming`, `agent_loop`. Not yet run against GeForce NOW.
- **2026-10-26**: Björn's GFN run (Olden Era, DeepSeek V4.1 Flash via OpenRouter, Gamer, 30-minute limit): streaming detected on step 1, screen capture of the 3840x2189 client, 0 blocked inputs, ~14 s per step. The time limit stopped the run and wrote the report; an earlier run's GFN window was closed 15 minutes after its limit.
