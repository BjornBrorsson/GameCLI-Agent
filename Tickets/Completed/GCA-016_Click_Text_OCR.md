# GCA-016 — click_text: press a button by its visible label (OCR)

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
Menu buttons with readable labels are the easiest targets to name but were still clicked by estimated coordinates. `click_text <label>` finds the label with Tesseract and clicks its centre; game agnostic (pure vision).

## Acceptance Criteria
- [x] Finds menu labels exactly (SANDBOX, LOAD GAME, Rest, Smith in tests; ~1 s per lookup)
- [x] Falls back with a clear 'not found' outcome so the model clicks by coordinates instead
- [x] OCR-located clicks skip template drift correction
- [ ] Small boxed labels (e.g. ALERTS, lift cards) are not readable by Tesseract — known limit

## Work Log
- **2026-10-25**: Filed and built by Claude Code after Björn asked to focus on computer use; game agnostic (vision, reasoning and computer use only).
- **2026-10-25**: Built: `backend/text_locator.py`. Live run on Ski Resort Tycoon (OpenRouter Haiku 5.5, Reviewer, 42 steps, 10.5 min, $0.19): built a path, a lift and a building and finished 5 of 7 goals; 62% of steps made progress (first review run: 38%, 43 Escape presses vs 9 now). click_text hit 2 of 4 (SANDBOX, Details...), the misses were icon-only buttons without visible text.
