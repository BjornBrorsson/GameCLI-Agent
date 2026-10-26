# GCA-003 — Direct Anthropic API provider with a fast model and prompt caching

| Field | Value |
|-------|-------|
| **Epic** | Speed |
| **Type** | Feature |
| **Priority** | High |
| **Status** | Assistance Required |
| **Assignee** | Claude Code |
| **Estimate** | M |
| **Depends on** | — |
| **Labels** | `llm`, `performance` |

## Summary
With `claude_code_cli` each step took 17-25 s: every call starts a full Claude Code agent that reads the screenshot and prompt files with tools (`backend/llm.py:466-554`), and a step can make three such calls (plan, retry assist, revalidate). A direct Messages API call with a fast model should take roughly 1-3 s (inferred, not measured).

## Requirements & Scope
- New provider `anthropic_api` in `llm.py` using the official `anthropic` SDK, image passed inline as base64, default model `claude-haiku-5-5`, configurable.
- Prompt caching on the static system prompt, role text, tools and memories (`cache_control` on the last static block).
- Merge plan + verify + revalidate into one call per step (works with GCA-001).
- Structured output via a tool definition (`emit_actions`) instead of free-text JSON parsing.
- Track real token usage and cost from the response; add Claude models to `MODEL_PRICING`.
- API key from `.env` / the dashboard, never written to logs.

## Acceptance Criteria
- [ ] Median step latency under 4 s on the Ski Resort Tycoon instructions (measured, logged)
- [ ] Cache hits visible in usage after the first step
- [x] Provider selectable in the dashboard
- [x] No change in behaviour for existing providers

## Work Log
- **2026-10-25**: Filed by Claude Code from the Ski Resort Tycoon playtest (https://claude.ai/artifact/LBzaXnMhoGCDQhQ5VkZxeQ).
- **2026-10-25**: Built (Claude Code). New `anthropic_api` provider using the official `anthropic` SDK (added to requirements): default `claude-haiku-5-5`, effort `low`, screenshot inline, static prompt in a cached system block, structured output via `output_config.format` (STEP_SCHEMA), refusal handling, cache-aware cost tracking. Uses the single combined call per step. Dashboard: new provider with an optional key field.
- **2026-10-25**: Unit-tested with a fake client (request shape, cache split, cost). NOT live-tested: no ANTHROPIC_API_KEY or `ant auth login` profile on this PC. Needs a key to measure the < 4 s step latency and cache hits. (The Gemini key in `.env` returns 402 Payment Required, so the Gemini API path could not be used for a live run either.)
