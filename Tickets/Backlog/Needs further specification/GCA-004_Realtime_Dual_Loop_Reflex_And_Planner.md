# GCA-004 — Real-time mode: fast reflex loop plus slow planner

| Field | Value |
|-------|-------|
| **Epic** | Real-time play |
| **Type** | Architecture |
| **Priority** | Normal |
| **Status** | Backlog |
| **Assignee** | — |
| **Estimate** | L |
| **Depends on** | GCA-001, GCA-002, GCA-003 |
| **Labels** | `agent-loop`, `realtime` |

## Summary
The current loop (capture → one LLM call → execute → wait) cannot keep up with games that move without the player. Split it in two:
- **Reflex loop**: fast model, 1-3 Hz, small frame (e.g. 640×360) plus the current sub-goal and the last action's result; emits short action bursts (held keys with durations, aim/move, clicks).
- **Planner**: stronger model every 20-30 s or on events (new window, goal reached, repeated no-progress); writes the current sub-goal and constraints the reflex loop follows.

## Requirements & Scope
- Input layer: `hold key for N ms`, simultaneous keys, mouse move with velocity; non-blocking execution so capture continues during actions.
- Event detection between frames (pHash jump, new modal) to wake the planner early.
- Option for turn-based/tycoon games to auto-pause while the planner thinks.
- Pick a simple real-time test game and define a measurable success (e.g. survive N seconds, reach a checkpoint).

## Acceptance Criteria
- [ ] Design note agreed (loop rates, message format between planner and reflex loop)
- [ ] Test game and success metric chosen
- [ ] Prototype runs the reflex loop at ≥ 1 Hz end to end

## Work Log
- **2026-10-25**: Filed by Claude Code from the Ski Resort Tycoon playtest (https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ).
