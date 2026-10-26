# GameCLI-Agent — Plan

## Vision
A game-agnostic agent that watches a game window and plays it through mouse and keyboard, reliably enough to playtest games (QA + review) and, with a fast model, to play real-time games.

## Current state (2026-10-25)
A playtest of Ski Resort Tycoon (54 steps, Claude Code CLI + Haiku) showed the agent decides sensibly but loses most of its time to false failures: a pixel-diff success check, coordinate nudges onto neighbouring buttons, and a pause on every failed action. Each step takes 17-25 s because each LLM call starts a full CLI agent. Full report: https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ

## Epics
1. **Reliable actions** — model-judged success, safe capture, no blind retries (GCA-001, GCA-002, GCA-007, GCA-008).
2. **Speed** — direct API provider with a fast model (GCA-003).
3. **Real-time play** — fast reflex loop + slow planner (GCA-004).
4. **Game adapters** — click by element name and read state when a game exposes it (GCA-005, Ski Resort Tycoon SKTY-157).
5. **Memory** — game-scoped facts instead of pixel coordinates (GCA-006).

## Suggested order
GCA-002 → GCA-001 → GCA-007 → GCA-008 → GCA-003 → GCA-006 → GCA-005 → GCA-004
