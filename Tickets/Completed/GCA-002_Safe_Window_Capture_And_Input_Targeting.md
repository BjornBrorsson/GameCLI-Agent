# GCA-002 — Capture the game's client area only and never send input to another window

| Field | Value |
|-------|-------|
| **Epic** | Reliable actions |
| **Type** | Bug |
| **Priority** | Critical |
| **Status** | Completed |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `capture`, `input`, `safety` |

## Summary
Three problems seen in the playtest:
1. `mss` grabs screen pixels inside the window rect, so when the Claude app and later an IDE came to the front, the agent read them as the game and clicked into them three times.
2. The capture includes the title bar and borders (1316×823 at 300% DPI) and is squashed to 1280×720, a 14% vertical distortion, while the prompt says the image is the game's 1280×720.
3. `getWindowsWithTitle(name)[0]` is a substring match, and on no match it silently falls back to the whole monitor (`screen_capture.py:76, 93`).

The working tree has an uncommitted `PrintWindow` fallback (`_capture_window_gdi`) that is used only when the `mss` frame is all black; it is a good start for point 1.

## Requirements & Scope
- Capture the client area: `GetClientRect` + `ClientToScreen` in a per-monitor DPI-aware process, offsets = client origin. Use `PrintWindow(PW_RENDERFULLCONTENT)` or Windows Graphics Capture as the primary path so covering windows don't matter.
- Keep the true aspect ratio: scale to fit 1280 wide (or send native size) and tell the model the real width and height.
- Before every input, check `WindowFromPoint(target)`'s root window is the game hwnd; if not, re-focus once, and if still covered skip the input and report it.
- Match the window title exactly (or by hwnd chosen in the UI); stop with an error instead of falling back to the monitor.
- Detect window size/state changes (maximize, move) between capture and input and re-capture.

## Acceptance Criteria
- [x] With another window covering the game, the agent sends no input to the other window
- [x] Screenshots are the client area without title bar, at the correct aspect ratio
- [x] Wrong or missing window title stops the run with a clear error
- [ ] Maximizing the game mid-run does not cause misplaced clicks

## Work Log
- **2026-10-25**: Filed by Claude Code from the Ski Resort Tycoon playtest (https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ).
- **2026-10-25**: Built (Claude Code). Capture = client area via PrintWindow(PW_CLIENTONLY|PW_RENDERFULLCONTENT) with an mss fallback, aspect ratio kept (`screen_capture.py`). `find_window` matches the exact title, or a unique substring, and raises `WindowNotFoundError` instead of falling back to the monitor; the loop stops on it. Every mouse input checks `WindowFromPoint` is the game (one foreground retry); keys go via SendInput only while the game is foreground, otherwise a single PostMessage — fixes the uncommitted change that sent q/e/m/space/esc twice (`input_controller.py`). Window move/resize between capture and input aborts the remaining actions and replans.
- **2026-10-25**: Verified: click/hover at a point outside the game returns `blocked: … covered by 'Claude'` with nothing sent; live capture of the 1280x720 client area. Not live-tested: maximizing mid-run (logic is the region comparison in `agent_loop.py`).
