import re
import shutil
import subprocess
import json
import base64
import os
import tempfile
import time

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None

# Direct Anthropic Messages API (official SDK), the fast path for real-time play.
ANTHROPIC_PROVIDER = "anthropic_api"

SYSTEM_PROMPT = """
You are an autonomous AI playing a video game via screenshot analysis and synthetic input.

INPUT: A screenshot of the current game state + the user's Game Instructions.
The screenshot has YELLOW COORDINATE RULERS on all four edges — use these to measure positions precisely.
OUTPUT: JSON only. No markdown, no backticks, no text outside the JSON.

FORMAT:
{
  "narration": "Brief calm analysis: what you see, your options, why you chose these actions. Internal monologue only. No greetings, hype, sign-offs, or exclamation marks.",
  "actions": [
    {"command": "press enter", "reason": "Confirm selection using keyboard", "precondition": "Phase == Combat", "wait_after_condition": "enemy_turn_animation_ends"},
    {"command": "click 1450 780", "reason": "Click the End Turn button"}
  ]
}

{role_instructions}

Each action is an object with "command" (the input command) and "reason" (short explanation of what this action does and why).
You can optionally include "precondition" to assert a game state (e.g. "Phase == Combat") before the action executes, and "wait_after_condition" to instruct the agent to pause until a visual animation resolves.

COMMANDS (prefer keyboard over mouse when possible):
- press key — single key tap (enter, escape, space, tab, e, 1, 2, f5, etc). PREFERRED for buttons and shortcuts.
- hold_key / release_key key — modifier hold/release (shift, ctrl, alt). Always pair them.
- type text — type a string character by character.
- click / right_click / middle_click / double_click x y — use when no keyboard shortcut exists.
- drag x1 y1 x2 y2 — click-hold, move, release. For draggable items, sliders, camera pan.
- scroll x y amount — mouse wheel (positive=up, negative=down). For zoom, lists.
- hover x y — move cursor without clicking. For tooltips, inspection.
- wait seconds — pause (decimal ok, e.g. 0.5). ONLY when the game needs time for animations.
- run_macro macro_name — run a pre-recorded sequence of actions (e.g., standard opening combat turns or menu navigation).

KEYBOARD SHORTCUTS — USE THEM:
- ALWAYS prefer keyboard shortcuts over clicking when they exist.
- Many games have hotkeys: Space/Enter=confirm, Escape=cancel/back, number keys=select items, letter keys=actions.
- Look for underlined letters, key hints in brackets [K], or tooltip hotkey indicators.
- If you know a game's shortcuts from your training data, use them.
- If a shortcut doesn't work, the system will retry — fall back to clicking only after keyboard fails.

COORDINATE PRECISION:
- The screenshot is always 1280×720 pixels. All coordinates are in this space.
- The screenshot has yellow rulers on all four edges with tick marks every 50px and labels every 100px.
- Use these rulers to measure the EXACT center of the element you want to interact with.
- For drag commands: x1,y1 must be the EXACT CENTER of the source element, x2,y2 must be on the drop target.
- Common mistake: estimating coordinates without checking against the rulers. Always cross-reference.
- UI elements may shift position after interactions — always re-measure using the rulers.

RULES:
- Return ONLY valid JSON.
- Minimize wait commands — the system already pauses between actions.
- Keep narration concise (2-4 sentences max).
"""

# ── Role definitions ──
# Each role provides instructions that modify the agent's behaviour and narration style.
ROLE_PROMPTS = {
    "gamer": (
        "ROLE: Gamer — You are playing this game to win and have a good time.\n"
        "- Play skillfully and make smart strategic decisions.\n"
        "- Narrate your thought process naturally, like a player thinking out loud.\n"
        "- Take calculated risks when the payoff is worth it.\n"
        "- Celebrate clever plays briefly, but stay focused on the game."
    ),
    "reviewer": (
        "ROLE: Game Reviewer — You are evaluating this game for a critical review.\n"
        "- Play normally but pay close attention to game design, UX, mechanics, difficulty balance, and polish.\n"
        "- In your narration, note both positives and negatives you observe (controls, feedback, readability, pacing).\n"
        "- Comment on how intuitive the UI is, whether instructions are clear, and if mechanics feel fair.\n"
        "- Periodically include a brief evaluative note, e.g. 'The card tooltip placement obscures the board here.'\n"
        "- Still play to make progress — a reviewer needs to experience the game, not stall.\n"
        "- Record what you experience as findings (positives too). Do not spend steps reproducing problems;\n"
        "  if a problem might be your own input mistake, say so in the finding and set confirmed=false."
    ),
    "tester": (
        "ROLE: QA Tester — You are testing this game for bugs, glitches, and edge cases.\n"
        "- Systematically explore the UI: try hovering elements, clicking unexpected areas, using wrong inputs.\n"
        "- Attempt actions that might break things: spam clicks, interact during animations, use items in odd contexts.\n"
        "- In your narration, log anything unusual: visual glitches, misaligned elements, unresponsive buttons, text overflow.\n"
        "- Note the reproduction steps for any issue you find.\n"
        "- Alternate between normal gameplay and deliberate edge-case probing.\n"
        "- Finding bugs is the goal. When something looks like a bug, first rule out your own input mistake:\n"
        "  reproduce it once with a different input (keyboard instead of mouse, clicking the exact centre).\n"
        "  Report it with confirmed=true only after it reproduced; otherwise confirmed=false.\n"
        "- Record every issue as a finding with numbered repro steps in its details."
    ),
    "speedrunner": (
        "ROLE: Speedrunner — You are trying to complete the game as fast as possible.\n"
        "- Minimize time and unnecessary actions. Skip optional content, dialogue, and animations.\n"
        "- Prefer keyboard shortcuts over mouse clicks for speed.\n"
        "- Take the most efficient path — avoid exploration that doesn't advance the objective.\n"
        "- In your narration, briefly note your routing decisions and time-saving choices.\n"
        "- Accept higher risk if it saves significant time."
    ),
    "explorer": (
        "ROLE: Explorer — You are mapping out unknown areas and looking for secrets.\n"
        "- Prioritize checking unrevealed map boundaries, doors, and interactive objects.\n"
        "- Avoid combat if possible. If dragged into combat, try to flee or finish it quickly to resume exploring.\n"
        "- Be highly tolerant of random movements or walking into walls to test bounds.\n"
        "- Ignore optimization or primary quest objectives temporarily to focus on mapping."
    ),
}

DEFAULT_ROLE = "gamer"

REVALIDATE_PROMPT = """
You are a coordinate correction assistant for a game automation agent.

The game state changed after executing some actions. Analyze the CURRENT
screenshot and provide updated action commands for the remaining planned actions.
The screenshot has YELLOW COORDINATE RULERS on all four edges — use them for precise measurement.

IF THE SCREEN SHOWS THE NORMAL GAME STATE:
- Update coordinates for the remaining actions to match where elements are NOW
- UI elements shift after interactions (e.g. items reflow, units move) — re-measure positions using the rulers
- Targets may have been destroyed, removed, or changed position
- Remove actions targeting things that no longer exist

IF THE SCREEN SHOWS A PROMPT, DIALOG, OR SELECTION SCREEN:
- First return action(s) to handle the prompt (e.g. click to confirm, select an option)
- Then return the remaining planned actions with updated coordinates

COMMANDS (prefer keyboard over mouse when possible):
{commands_list_short}
- run_macro macro_name

RULES:
- Use the yellow rulers to measure coordinates precisely — do NOT estimate
- Return ONLY valid JSON, no markdown, no extra text
- Keep action reasons/descriptions where applicable

FORMAT:
{
  "actions": [
    {"command": "click 750 400", "reason": "Select the target unit", "precondition": "Phase == Combat", "wait_after_condition": "unit_selection_animation"},
    {"command": "press enter", "reason": "Confirm action using keyboard"}
  ]
}
"""

RETRY_ASSIST_PROMPT = """
You are a game automation assistant debugging a failed action.

The agent tried to execute this action multiple times but the screen never changed:
  Action: {failed_action}
  Reason: {failed_reason}
  Attempts: {attempts}

The attached screenshot shows the CURRENT game state.
It has YELLOW COORDINATE RULERS on all four edges — use them to measure positions precisely.

Analyze the screenshot and decide:
1. If a KEYBOARD SHORTCUT could achieve the same goal → use "press key" instead (strongly preferred)
2. If the action's INTENT is still valid but coordinates are wrong → use the rulers to provide corrected coordinates
3. If a different action would achieve the same goal → provide that instead
4. If the action is impossible (e.g. not enough resources, target doesn't exist) → respond with "skip"

Common issues:
- Coordinates were estimated wrong — use the yellow rulers to measure precisely
- A keyboard shortcut exists that avoids coordinate issues entirely
- UI elements shifted position after a previous interaction
- A dialog/popup appeared that needs to be dismissed first
- The action is simply not possible in the current game state

FORMAT (JSON only, no markdown, no backticks):
{{
  "command": "press enter",
  "reason": "Using keyboard shortcut instead of clicking the button"
}}
OR with corrected coordinates:
{{
  "command": "click 580 870",
  "reason": "Corrected: the button center is at x=580 based on the ruler markings"
}}
OR to skip:
{{
  "command": "skip",
  "reason": "This action is not possible because..."
}}
"""

LEAD_AGENT_PROMPT = """
You are the Lead Agent — a strategic analyst for an autonomous game-playing system.

{role_instructions}

INPUT: A screenshot of the current game state with YELLOW COORDINATE RULERS on all four edges.
OUTPUT: A concise strategic directive in plain text (NOT JSON).

Your job is to:
1. IDENTIFY THE GAME. State which game this is (e.g. "Slay the Spire", "Civilization VI",
   "Stardew Valley"). If you recognise it, leverage your knowledge of the game's mechanics,
   UI conventions, and optimal strategies. If you are unsure, say so and play generically.
2. Describe what you see on screen (game phase, important UI elements, threats, opportunities).
3. Decide on the best high-level strategy for this moment.
4. Give a clear directive to the Worker Agent, including:
   - What actions to take and in what order
   - Which screen elements to interact with (describe their approximate position)
   - KEYBOARD SHORTCUTS: Always prefer keyboard hotkeys over mouse clicks when possible.
     If you know the game's hotkeys (e.g. number keys for cards, E for end turn, Enter to
     confirm), instruct the Worker to use them. They are faster and more reliable than clicks.
   - If you discover a useful hotkey or game mechanic, instruct the Worker to save it with
     save_memory so it persists across sessions.
   - Warnings about risks or things to avoid

Keep your directive concise (3-8 sentences). Focus on WHAT to do, not HOW to format it.
"""

# Shared by the Worker and Combined prompts: the agent judges whether its
# previous actions worked from the new screenshot (GCA-001) instead of the
# harness guessing from pixel diffs.
STEP_CHECK_RULES = """
CHECKING YOUR PREVIOUS STEP:
- The request may include "PREVIOUS STEP", listing the actions you sent last time, what you expected each one to do,
  and how the harness ran them (ok / blocked / screen changed or not).
- Look at the CURRENT screenshot and say in "last_step_check" which expectations came true and which did not.
  A hover, tooltip or button highlight counts as success when it is visible now, even if the screen barely changed.
- "progress": true if the game moved closer to the current goal since the previous step, false if it did not.
- Do not repeat an action that already did what you wanted. If an action failed, try a different approach.
- Give every action an "expect": the visible result you expect (e.g. "Path tool highlighted, tooltip shown").

GOALS AND FINDINGS:
- The request may include a GOALS checklist. Work on the CURRENT goal. Set "goal_done": true only when the
  screenshot shows the current goal is complete.
- The checklist was planned before the game was visible and can be wrong. If the current or upcoming goals do
  not fit the game you now see (mechanics that don't exist here, a different objective shown on screen), put a
  corrected list of the REMAINING goals in "goals_update" (3-8 goals, each one checkable thing). Otherwise [].
- "findings": things worth reporting from this step (can be empty). Each finding has a type
  (bug / ux / positive / note), a severity (low / medium / high), a short title, details, and "confirmed".
  Only record something new and notable — at most 2 findings per step, usually 0. Never repeat anything in the
  RECORDED FINDINGS list.

RUN NOTEBOOK:
- The request includes a RUN NOTEBOOK: facts you learned earlier in THIS run. Read it before planning and follow it.
- Put new facts in "notes_add" (0-3 short sentences per step): controls that worked or failed ("units move with
  two clicks: first shows the path, second confirms"), game rules, and what the game's own messages said
  ("Cannot recruit: a hero is already in this town"). Copy rule-like game messages almost word for word.
  Never add pixel coordinates, and never repeat a note that is already in the notebook.

BEFORE ACTING:
- "actor_now": in turn-based games, say who acts right now and with what (e.g. "my turn — Centaurs (active stack)",
  "enemy turn — wait"). Plan only for the actor that is actually active. Use "" for games without turns.
- When you are not sure what a click will do, check what the game says it will do first: hover the target and read
  the status line, tooltip or cursor shape, or open its info panel (often right-click). Then act.

AIMING:
- Red numbered crosshairs on the screenshot (if any) show where your previous step's mouse actions landed;
  the number is the action's number in PREVIOUS STEP. If one missed its target, correct by that offset.
- If a click missed twice, use click_text for labelled buttons, or zoom first and click from the zoomed view.

EFFICIENCY:
- Hover only when you need to read a tooltip; do not hover before every click.
- Avoid "wait": the harness already lets the screen settle after each action. Waits are capped at 2 seconds.
- Plan several actions per step when the screen will not change in between (e.g. play a whole turn).
"""

WORKER_AGENT_PROMPT = """
You are the Worker Agent — a precise input-command generator for an autonomous game-playing system.

INPUT: A screenshot with YELLOW COORDINATE RULERS on all edges, plus a strategic directive from the Lead Agent.
OUTPUT: JSON only. No markdown, no backticks, no text outside the JSON.

FORMAT:
{
  "last_step_check": "Which of the previous step's expectations are visible now and which are not.",
  "progress": true,
  "goal_done": false,
  "goals_update": [],
  "notes_add": ["Short new fact learned this step"],
  "actor_now": "",
  "findings": [{"type": "ux", "severity": "low", "title": "Short title", "details": "What happened", "confirmed": false}],
  "narration": "Brief description of what you're doing and why.",
  "actions": [
    {"command": "press enter", "reason": "Confirm selection", "expect": "Dialog closes"},
    {"command": "click 750 400", "reason": "Click the target element", "expect": "Element turns highlighted"}
  ]
}
""" + STEP_CHECK_RULES + """

COMMANDS (prefer keyboard over mouse when possible):
- press key — single key tap (enter, escape, space, tab, e, 1, 2, f5, etc).
- hold_key / release_key key — modifier hold/release (shift, ctrl, alt). Always pair them.
- type text — type a string character by character.
- click / right_click / middle_click / double_click x y — use when no keyboard shortcut exists.
- drag x1 y1 x2 y2 — click-hold, move, release.
- scroll x y amount — mouse wheel (positive=up, negative=down).
- hover x y — move cursor without clicking.
- click_text label — click the visible text "label" (button or menu item); the harness finds it on screen with OCR.
  Most reliable way to press labelled buttons. Falls back to nothing if the text can't be read — then use click.
- zoom x1 y1 x2 y2 — see that region enlarged in the NEXT screenshot (labels stay in full-screen coordinates).
  Use it on a SMALL area (under 480 px wide) before clicking small or crowded targets, or when a click missed. Put it last in the step.
- wait seconds — pause (decimal ok). ONLY when the game needs time.
- run_macro macro_name — run a pre-recorded sequence.
- save_memory text — save a useful discovery (hotkey, strategy, game mechanic) for future sessions.
  Example: save_memory In Slay The Spire, press 1-9 to select cards. Press E to end turn.
  Save facts, never pixel coordinates (they go stale when windows move).
  To correct an earlier memory, start with its id: save_memory replaces=<id> corrected text.

COORDINATE PRECISION:
- Coordinates are pixels in the screenshot; its exact size is given in the request.
- Use the yellow rulers to measure the EXACT center of elements. Always cross-reference.

RULES:
- Return ONLY valid JSON.
- Follow the Lead Agent's directive precisely.
- Use keyboard shortcuts whenever possible.
- Keep narration concise (2-4 sentences max).
"""

# ── Combined single-call prompt for CLI mode (avoids 2nd subprocess) ──
COMBINED_AGENT_PROMPT = """
You are an autonomous game-playing AI that analyzes screenshots and generates precise input commands.

{role_instructions}

INPUT: A screenshot of the current game state with YELLOW COORDINATE RULERS on all four edges.
OUTPUT: Valid JSON only. No markdown, no backticks, no text outside the JSON.

STEP 1 — CHECK: Compare the screenshot with what your previous actions were expected to do.
STEP 2 — ANALYZE: Identify the game, describe the current state, and decide on a strategy.
STEP 3 — GENERATE COMMANDS: Produce precise input commands to execute that strategy.

FORMAT:
{
  "last_step_check": "Which of the previous step's expectations are visible now and which are not.",
  "progress": true,
  "goal_done": false,
  "goals_update": [],
  "notes_add": ["Short new fact learned this step"],
  "actor_now": "",
  "findings": [{"type": "ux", "severity": "low", "title": "Short title", "details": "What happened", "confirmed": false}],
  "narration": "What game this is, what you see, and what you plan to do (2-5 sentences).",
  "actions": [
    {"command": "press enter", "reason": "Confirm selection", "expect": "Dialog closes"},
    {"command": "click 750 400", "reason": "Click the target element", "expect": "Element turns highlighted"}
  ]
}
""" + STEP_CHECK_RULES + """
COMMANDS (prefer keyboard over mouse when possible):
- press key — single key tap (enter, escape, space, tab, e, 1, 2, f5, etc).
- hold_key / release_key key — modifier hold/release (shift, ctrl, alt). Always pair them.
- type text — type a string character by character.
- click / right_click / middle_click / double_click x y — use when no keyboard shortcut exists.
- drag x1 y1 x2 y2 — click-hold, move, release.
- scroll x y amount — mouse wheel (positive=up, negative=down).
- hover x y — move cursor without clicking.
- click_text label — click the visible text "label" (button or menu item); the harness finds it on screen with OCR.
  Most reliable way to press labelled buttons. Falls back to nothing if the text can't be read — then use click.
- zoom x1 y1 x2 y2 — see that region enlarged in the NEXT screenshot (labels stay in full-screen coordinates).
  Use it on a SMALL area (under 480 px wide) before clicking small or crowded targets, or when a click missed. Put it last in the step.
- wait seconds — pause (decimal ok). ONLY when the game needs time.
- run_macro macro_name — run a pre-recorded sequence.
- save_memory text — save a useful discovery (hotkey, strategy, game mechanic) for future sessions.
  Save facts, never pixel coordinates (they go stale when windows move).
  To correct an earlier memory, start with its id: save_memory replaces=<id> corrected text.

COORDINATE PRECISION:
- Coordinates are pixels in the screenshot; its exact size is given in the request.
- Use the yellow rulers to measure the EXACT center of elements. Always cross-reference.

RULES:
- Return ONLY valid JSON.
- KEYBOARD SHORTCUTS: Always prefer keyboard hotkeys over mouse clicks when possible.
  If you know the game's hotkeys, use them — they are faster and more reliable than clicks.
- If you discover a useful hotkey or game mechanic, include a save_memory action.
- Keep narration concise.
"""

# ── Pricing per 1M tokens (input, output) in USD ──
MODEL_PRICING = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-flash-preview": (0.15, 0.60),
    "gemini-3.1-pro-preview": (1.25, 10.00),
    "anthropic/claude-sonnet-4-20250514": (3.00, 15.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "google/gemini-2.5-flash-preview": (0.15, 0.60),
    "google/gemini-2.5-pro-preview": (1.25, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# Cache reads/writes relative to the input price (Anthropic API).
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25

# Models offered for the direct Anthropic API provider (fastest first).
ANTHROPIC_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

# JSON schema for one planning step (structured output on the Anthropic API).
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "last_step_check": {"type": "string"},
        "progress": {"type": "boolean"},
        "goal_done": {"type": "boolean"},
        "goals_update": {"type": "array", "items": {"type": "string"}},
        "notes_add": {"type": "array", "items": {"type": "string"}},
        "actor_now": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["bug", "ux", "positive", "note"]},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "title": {"type": "string"},
                    "details": {"type": "string"},
                    "confirmed": {"type": "boolean"},
                },
                "required": ["type", "severity", "title", "details", "confirmed"],
                "additionalProperties": False,
            },
        },
        "narration": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "reason": {"type": "string"},
                    "expect": {"type": "string"},
                },
                "required": ["command", "reason", "expect"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["last_step_check", "progress", "goal_done", "goals_update", "notes_add", "actor_now",
                 "findings", "narration", "actions"],
    "additionalProperties": False,
}
DEFAULT_PRICING = (1.00, 3.00)  # conservative fallback

# Stronger model used for a step when the agent is stuck (GCA-009).
ESCALATION_MODELS = {
    "anthropic_api": "claude-sonnet-5-5",
    "claude_code_cli": "sonnet",
    "openrouter": "anthropic/claude-sonnet-5.5",
}


def escalation_model(provider: str, model_name: str) -> str:
    """Model to plan with when stuck; the current model when none is known
    or it already is the stronger one."""
    return ESCALATION_MODELS.get(provider, model_name)


GOAL_PLAN_PROMPT = """
You are planning a play session for an autonomous game-playing agent.
{role_instructions}

Split the user's instructions below into 3 to 8 ordered goals the agent can check off one at a time,
using the attached screenshot for context. Each goal must be concrete and visibly checkable on screen.
Each goal is ONE thing: when the instructions list several things to do or build ("paths, a lift and a
building"), make each of them its own goal. Do not add goals for narrating or note-taking — the agent
always does that.
Keep the user's own wording where possible. If the instructions are a single long objective
(e.g. "win a run"), split it into its natural stages.

Return JSON only: {{"goals": ["first goal", "second goal"]}}

USER INSTRUCTIONS:
{instructions}
"""

REPORT_PROMPTS = {
    "tester": "Write a QA bug report. Group confirmed bugs (reproduced) and unconfirmed issues separately, "
              "each with severity, numbered repro steps, expected vs actual, and the screenshot file name. "
              "Then list UX issues, then goal coverage (what was tested, what was skipped and why).",
    "reviewer": "Write a mock review for a games website: title, one-line verdict, score out of 10, then sections "
                "First impressions, Gameplay, Interface and controls, Look and feel, What needs work, Verdict. "
                "600-900 words. Say plainly where something may have been your own input mistake.",
    "gamer": "Write a short run summary: outcome, key decisions and turning points, what went well, what went wrong, "
             "and what to do differently next time.",
    "speedrunner": "Write a short speedrun summary: outcome, route taken, time and steps used, time losses, "
                   "and route improvements.",
    "explorer": "Write an exploration log: areas and systems discovered, secrets or oddities, and what remains unexplored.",
}

REPORT_PROMPT = """
You just finished an automated play session of a game. Write the report below in Markdown, based ONLY on the
session data. Do not invent anything that is not in the data.

REPORT TO WRITE: {report_kind}

USER INSTRUCTIONS FOR THE SESSION:
{instructions}

{session_data}
"""

# Providers that shell out to a local, pre-authenticated CLI binary.
# No API key and no `requests` dependency required — each CLI uses its
# own stored login (e.g. ~/.claude credentials, agy account, gemini auth).
CLI_PROVIDERS = frozenset({"gemini_cli", "claude_code_cli", "antigravity_cli"})


def is_cli_provider(provider: str) -> bool:
    return provider in CLI_PROVIDERS


def find_cli_binary(*names: str):
    """Resolve a CLI binary: PATH first, then well-known install locations."""
    for name in names:
        exe = shutil.which(name)
        if exe:
            return exe
    home = os.path.expanduser("~")
    well_known = [
        os.path.join(home, ".local", "bin", "claude.exe"),
        os.path.join(home, ".local", "bin", "claude"),
        os.path.join(os.environ.get("LOCALAPPDATA") or home, "agy", "bin", "agy.exe"),
    ]
    wanted = {n.lower() for n in names}
    for cand in well_known:
        stem = os.path.splitext(os.path.basename(cand))[0].lower()
        if stem in wanted and os.path.exists(cand):
            return cand
    return None


class BudgetExceededException(Exception):
    """Raised when the session budget limit is reached."""
    pass


class CostTracker:
    """Tracks token usage and estimated cost for API-based providers."""

    def __init__(self, max_budget_usd: float = 0.0):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.cache_read_tokens = 0
        self.call_count = 0
        self.max_budget_usd = max_budget_usd

    def record(self, input_tokens: int, output_tokens: int, model_name: str,
               cache_read_tokens: int = 0, cache_write_tokens: int = 0):
        # Prevent API calls if we are already at or over budget
        if self.max_budget_usd > 0.0 and self.total_cost_usd >= self.max_budget_usd:
            raise BudgetExceededException(f"Budget exceeded: {self.total_cost_usd:.4f} >= {self.max_budget_usd:.4f}")

        self.total_input_tokens += input_tokens + cache_read_tokens + cache_write_tokens
        self.total_output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.call_count += 1
        pricing = MODEL_PRICING.get(model_name, DEFAULT_PRICING)
        cost = (input_tokens * pricing[0]
                + cache_read_tokens * pricing[0] * CACHE_READ_FACTOR
                + cache_write_tokens * pricing[0] * CACHE_WRITE_FACTOR
                + output_tokens * pricing[1]) / 1_000_000
        self.total_cost_usd += cost

        # After updating, check if we crossed the limit
        if self.max_budget_usd > 0.0 and self.total_cost_usd > self.max_budget_usd:
            raise BudgetExceededException(f"Budget exceeded: {self.total_cost_usd:.4f} > {self.max_budget_usd:.4f}")

    def get_summary(self) -> dict:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "call_count": self.call_count,
        }


class LLMIntegration:
    PROVIDER_ENDPOINTS = {
        "gemini_api": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }

    def __init__(self, provider: str = "gemini_cli", api_key: str = None,
                 secondary_api_key: str = None, max_budget_usd: float = 0.0):
        self.provider = provider
        self.api_key = api_key
        self.secondary_api_key = secondary_api_key
        self._using_secondary = False  # tracks which key is active
        self.cost = CostTracker(max_budget_usd)
        self.turn_count = 0
        self.two_call = False  # Lead/Worker split for API providers (off: one call per step)
        self._extra_image = None  # optional second frame for the current planning call (GCA-025)
        self._anthropic = None
        if provider == ANTHROPIC_PROVIDER:
            if _anthropic_sdk is None:
                raise ImportError("'anthropic' package is required for the Anthropic API provider. "
                                  "Install with: pip install anthropic")
            # No key from the dashboard → the SDK resolves ANTHROPIC_API_KEY,
            # ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
            key = api_key if api_key and api_key != "local-cli" else None
            self._anthropic = (_anthropic_sdk.Anthropic(api_key=key, timeout=60.0) if key
                               else _anthropic_sdk.Anthropic(timeout=60.0))
        elif provider not in CLI_PROVIDERS and _requests is None:
            raise ImportError("'requests' package is required for API providers. "
                              "Install with: pip install requests")

    @staticmethod
    def _strip_thinking_tags(text: str) -> tuple:
        """Extract and remove <thought>...</thought>, <thinking>...</thinking>,
        and similar reasoning tags that some models emit before the actual content.
        Returns (cleaned_text, thinking_text) so thinking can be logged separately."""
        TAG_PATTERN = re.compile(
            r'<(thought|thinking|reasoning|inner_monologue)>(.*?)</\1>',
            flags=re.DOTALL)
        thinking_parts = [m.group(2).strip() for m in TAG_PATTERN.finditer(text)]
        cleaned = TAG_PATTERN.sub('', text).strip()
        thinking = "\n".join(thinking_parts) if thinking_parts else ""
        return cleaned, thinking

    # ── Routing ──

    def _call(self, prompt_text: str, image_base64: str, model_name: str, expect_json: bool = True):
        """Route to the configured provider."""
        if self.provider in CLI_PROVIDERS:
            return self._call_cli(prompt_text, image_base64, model_name, expect_json=expect_json)
        elif self.provider == ANTHROPIC_PROVIDER:
            return self._call_anthropic("", prompt_text, image_base64, model_name, expect_json=expect_json)
        else:
            return self._call_api(prompt_text, image_base64, model_name, expect_json=expect_json)

    # ── Anthropic API (direct, fast) ──

    def _call_anthropic(self, static_text: str, dynamic_text: str, image_base64: str,
                        model_name: str, expect_json: bool = True, schema: dict = None):
        """One Messages API call with the screenshot inline.

        static_text (system prompt, role, game instructions) goes in a cached
        system block so repeated steps only pay full price for the dynamic
        part (history, memories, previous step). `schema` turns on structured
        output so the reply is guaranteed to parse.
        """
        if self.cost.max_budget_usd > 0.0 and self.cost.total_cost_usd >= self.cost.max_budget_usd:
            raise BudgetExceededException(f"Budget exceeded: {self.cost.total_cost_usd:.4f} >= {self.cost.max_budget_usd:.4f}")

        kwargs = {}
        if static_text:
            kwargs["system"] = [{"type": "text", "text": static_text,
                                 "cache_control": {"type": "ephemeral"}}]
        output_config = {"effort": "low"}  # latency matters more than depth per step
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        t0 = time.time()
        response = self._anthropic.messages.create(
            model=model_name,
            max_tokens=8000,
            output_config=output_config,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": image_base64}},
                    *([{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": self._extra_image}}] if self._extra_image else []),
                    {"type": "text", "text": dynamic_text},
                ],
            }],
            **kwargs,
        )
        elapsed = time.time() - t0

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise ValueError(f"model declined the request (refusal, category={category})")
        content = next((b.text for b in response.content if b.type == "text"), "")

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        result = self._extract_json(content) if expect_json else content
        try:
            self.cost.record(usage.input_tokens, usage.output_tokens, model_name,
                             cache_read_tokens=cache_read, cache_write_tokens=cache_write)
        except BudgetExceededException as e:
            e.partial_result = result
            raise
        print(f"  [api] anthropic {model_name}  in={usage.input_tokens} cache_read={cache_read} "
              f"cache_write={cache_write} out={usage.output_tokens}  {elapsed:.1f}s  "
              f"session=${self.cost.total_cost_usd:.4f}")
        if response.stop_reason == "max_tokens":
            print("  [api] WARNING: response hit max_tokens")
        return result

    # ── Gemini CLI (free, default) ──

    # Fixed temp paths — avoids mkstemp overhead per call.
    # Overwritten each call; no accumulation of temp files.
    _cli_tmp_dir = os.path.join(os.path.dirname(__file__), "..", ".tmp")
    _cli_img_path = os.path.join(_cli_tmp_dir, "_screenshot.jpg")
    _cli_img2_path = os.path.join(_cli_tmp_dir, "_screenshot2.jpg")
    _cli_prompt_path = os.path.join(_cli_tmp_dir, "_prompt.txt")

    def _extract_json(self, text: str) -> dict:
        """Robustly extract JSON from a potentially messy model response.
        Handles markdown code blocks, conversational prefix/suffix, and trailing junk.
        """
        # 1. Clean up markdown code block markers
        clean_text = re.sub(r'```(?:json)?', '', text).strip()
        
        # 2. Find the first '{' and last '}'
        start_idx = clean_text.find('{')
        end_idx = clean_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            raise ValueError(f"No JSON object found in output. Raw output: {text[:300]}")
            
        json_str = clean_text[start_idx:end_idx + 1]
        
        # 3. Handle common LLM JSON errors (e.g. trailing commas, missing quotes in keys)
        # Attempt standard parse first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try some basic repairs if it fails
            # Remove trailing commas before closing braces/brackets
            repaired = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON even after repair. Error: {e}\nJSON string: {json_str[:300]}")

    def _call_cli(self, prompt_text: str, image_base64: str, model_name: str,
                  expect_json: bool = True, reuse_image: bool = False):
        """Call a local agentic CLI: Gemini CLI, Claude Code, or Antigravity (agy).
        Free (uses each CLI's own stored auth) but slower than a direct API.

        If reuse_image=True, skips writing the screenshot (assumes it was
        already written by a prior call in the same step).
        """
        if not re.match(r'^[\w\-\.]+$', model_name):
            raise ValueError(f"Invalid model name format: {model_name}")

        os.makedirs(self._cli_tmp_dir, exist_ok=True)
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Write image (skip if caller says it's already there)
        if not reuse_image:
            img_data = base64.b64decode(image_base64)
            if self._extra_image:
                with open(self._cli_img2_path, 'wb') as f2:
                    f2.write(base64.b64decode(self._extra_image))
            with open(self._cli_img_path, 'wb') as f:
                f.write(img_data)

        # Write prompt
        with open(self._cli_prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt_text)

        if self.provider == "gemini_cli":
            rel_img = os.path.relpath(self._cli_img_path, workspace_root)
            rel_prompt = os.path.relpath(self._cli_prompt_path, workspace_root)
            full_prompt = f"Analyze this image: @{rel_img}. Task instructions: @{rel_prompt}"
            if self._extra_image:
                full_prompt += f" Second frame, a moment later: @{os.path.relpath(self._cli_img2_path, workspace_root)}"
            cmd = ["gemini", "-p", full_prompt, "--model", model_name]
            use_shell = True
            timeout = 120
            stdin = "1\n"
        else:
            # Agentic CLIs (Claude Code, agy) read files with their own tools —
            # pass plain absolute paths and let the agent load them itself.
            img_abs = os.path.abspath(self._cli_img_path).replace("\\", "/")
            prompt_abs = os.path.abspath(self._cli_prompt_path).replace("\\", "/")
            full_prompt = (
                f"Use your file-reading tool to view the screenshot image at \"{img_abs}\" "
                + (f"and the second frame (taken a moment later) at \"{os.path.abspath(self._cli_img2_path)}\" "
                   if self._extra_image else "") +
                f"and to read the task instructions at \"{prompt_abs}\". "
                f"Then follow those instructions exactly. "
                f"Your final reply must contain ONLY the raw JSON object — no prose, no markdown."
            )
            stdin = None
            use_shell = False
            timeout = 300
            if self.provider == "claude_code_cli":
                binary = find_cli_binary("claude")
                if not binary:
                    raise RuntimeError(
                        "Claude Code CLI not found on PATH or ~/.local/bin. "
                        "Install it and log in once (e.g. `irm https://claude.ai/install.ps1 | iex`).")
                tail = ["-p", full_prompt, "--model", model_name,
                        "--dangerously-skip-permissions",
                        "--add-dir", self._cli_tmp_dir,
                        "--output-format", "text", "--max-turns", "8"]
                # npm-installed builds land as .cmd shims that need cmd.exe
                cmd = (["cmd", "/c", binary] if binary.lower().endswith((".cmd", ".bat"))
                       else [binary]) + tail
            elif self.provider == "antigravity_cli":
                binary = find_cli_binary("agy")
                if not binary:
                    raise RuntimeError(
                        "Antigravity CLI (agy) not found on PATH or %LOCALAPPDATA%\\agy\\bin.")
                cmd = [binary, "-p", full_prompt, "--model", model_name,
                       "--print-timeout", "240s",
                       "--dangerously-skip-permissions",
                       "--add-dir", self._cli_tmp_dir,
                       "--output-format", "text"]
            else:
                raise ValueError(f"Unknown CLI provider: {self.provider}")

        result = subprocess.run(
            cmd,
            input=stdin,
            cwd=workspace_root,
            # CLIs print UTF-8; the Windows default codepage turns "—" into "â€”"
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
            shell=use_shell,
            timeout=timeout
        )
        content = result.stdout.strip()
        content, thinking = self._strip_thinking_tags(content)
        if thinking:
            print(f"  [thinking] {thinking}")

        if not expect_json:
            return content

        return self._extract_json(content)

    # ── API providers (Gemini API / OpenRouter) ──

    def _call_api(self, prompt_text: str, image_base64: str, model_name: str, expect_json: bool = True,
                  schema: dict = None):
        """Call an OpenAI-compatible chat completions API.
        Works for both Gemini API and OpenRouter.
        """
        # Pre-check budget before incurring costs
        if self.cost.max_budget_usd > 0.0 and self.cost.total_cost_usd >= self.cost.max_budget_usd:
            raise BudgetExceededException(f"Budget exceeded: {self.cost.total_cost_usd:.4f} >= {self.cost.max_budget_usd:.4f}")

        endpoint = self.PROVIDER_ENDPOINTS[self.provider]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/GameCLI-Agent"
            headers["X-Title"] = "GameCLI Agent"

        body = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }},
                    # optional second frame, after the main one (GCA-025)
                    *([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._extra_image}"}}]
                      if self._extra_image else []),
                ]
            }],
            "max_tokens": 8000,
            "temperature": 0.7,
        }
        # Structured output where the provider supports it, so a model can't
        # answer with tool-call XML instead of the step JSON. Dropped again
        # if the provider rejects it (HTTP 400).
        if schema is not None and self.provider == "openrouter":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "agent_step", "strict": True, "schema": schema}}

        # Retry transient errors with exponential backoff
        RETRYABLE = {429, 500, 502, 503, 504}
        MAX_API_RETRIES = 3
        last_err = None
        data = None
        elapsed = 0

        # Determine which key to start with (stay on secondary once switched)
        active_key = self.api_key if not self._using_secondary else (self.secondary_api_key or self.api_key)

        for attempt in range(MAX_API_RETRIES + 1):
            headers["Authorization"] = f"Bearer {active_key}"
            try:
                t0 = time.time()
                resp = _requests.post(endpoint, headers=headers, json=body, timeout=120)
                elapsed = time.time() - t0
                resp.raise_for_status()
                data = resp.json()
                
                # Check for truncation
                finish_reason = data["choices"][0].get("finish_reason")
                if finish_reason == "length":
                    print(f"  [api] WARNING: Response truncated due to length (max_tokens={body['max_tokens']})")
                
                break
            except _requests.exceptions.HTTPError as e:
                last_err = e
                # ── Key fallback: on 429 with primary key, swap to secondary ──
                if (resp is not None and resp.status_code == 429
                        and not self._using_secondary
                        and self.secondary_api_key
                        and active_key != self.secondary_api_key):
                    print(f"  [api] 429 rate-limited on primary key — switching to secondary key")
                    active_key = self.secondary_api_key
                    self._using_secondary = True
                    continue  # retry immediately with secondary key
                if resp is not None and resp.status_code == 400 and "response_format" in body:
                    print("  [api] 400 with structured output — retrying without response_format")
                    body.pop("response_format")
                    continue
                if resp is not None and resp.status_code in RETRYABLE and attempt < MAX_API_RETRIES:
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    print(f"  [api] {resp.status_code} on attempt {attempt+1}, "
                          f"retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
            except _requests.exceptions.ConnectionError as e:
                last_err = e
                if attempt < MAX_API_RETRIES:
                    wait = 2 ** (attempt + 1)
                    print(f"  [api] Connection error on attempt {attempt+1}, "
                          f"retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise

        if data is None:
            raise last_err or RuntimeError("API call failed with no response")

        # Extract content
        content = data["choices"][0]["message"]["content"]
        content, thinking = self._strip_thinking_tags(content)
        if thinking:
            print(f"  [thinking] {thinking}")

        # Track tokens / cost
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)

        if not expect_json:
            try:
                self.cost.record(in_tok, out_tok, model_name)
            except BudgetExceededException as e:
                e.partial_result = content
                raise
            return content

        # Parse JSON from content first so we don't lose the result if budget is exceeded
        result_json = self._extract_json(content)

        try:
            self.cost.record(in_tok, out_tok, model_name)
        except BudgetExceededException as e:
            # Attach the successful result to the exception so the caller can still use it
            e.partial_result = result_json
            cost_so_far = self.cost.total_cost_usd
            print(f"  [api] {self.provider} {model_name}  "
                  f"{in_tok}+{out_tok} tok  {elapsed:.1f}s  "
                  f"session=${cost_so_far:.4f}")
            raise

        cost_so_far = self.cost.total_cost_usd
        print(f"  [api] {self.provider} {model_name}  "
              f"{in_tok}+{out_tok} tok  {elapsed:.1f}s  "
              f"session=${cost_so_far:.4f}")

        return result_json

    def get_next_action(self, image_base64: str, game_instructions: str, model_name: str = "gemini-3-flash-preview",
                        role: str = "gamer", grounding_text: str = "", enabled_tools: list = None,
                        step_history: str = "", memories_text: str = "", last_step: str = "",
                        image_size: tuple = None, goals_text: str = "", stuck_text: str = "",
                        view_note: str = "", extra_image_b64: str = None, notebook_text: str = ""):
        """Plan one step. `last_step` describes the previous step's actions,
        their expected results and how they ran, so the model can judge them
        against the new screenshot (GCA-001)."""
        self._extra_image = extra_image_b64
        try:
            self.turn_count += 1
            if notebook_text:
                goals_text = "\n\n".join(filter(None, [notebook_text, goals_text]))
            if extra_image_b64:
                view_note = "\n".join(filter(None, [view_note, (
                    "TWO FRAMES: you get two screenshots taken about 0.4 s apart (the first one has the rulers and "
                    "markers). Things that differ between them are animations — a blinking active unit, a pulsing "
                    "valid move, a moving cursor or an incoming effect. Use that to tell what is active right now.")]))
            size_line = (f"Screenshot size: {image_size[0]}x{image_size[1]} pixels.\n"
                         if image_size else "")
            role_instructions = ROLE_PROMPTS.get(role, ROLE_PROMPTS[DEFAULT_ROLE])

            # Load dynamic tools from tools.json if available (PR #11: Dynamic Tools Registry)
            extra_commands = ""
            tools_path = os.path.join(os.path.dirname(__file__), 'tools.json')
            if os.path.exists(tools_path):
                try:
                    with open(tools_path, 'r', encoding='utf-8') as f:
                        tools = json.load(f)
                    tool_descs = [
                        tool.get('description', '')
                        for name, tool in tools.items()
                        if enabled_tools is None or name in enabled_tools
                    ]
                    if tool_descs:
                        extra_commands = "\n" + "\n".join(tool_descs)
                except Exception:
                    pass

            # ── CLI fast-path: single combined call ──
            # The CLI spawns a subprocess per call, so merging Lead+Worker
            # into one call saves ~50% of the per-step latency.
            # One combined call per step for every provider unless the
            # two-call Lead/Worker split is switched on (GCA-013).
            if not self.two_call or self.provider in CLI_PROVIDERS or self.provider == ANTHROPIC_PROVIDER:
                return self._get_next_action_combined(
                    image_base64, game_instructions, model_name, role_instructions,
                    grounding_text, extra_commands, step_history, memories_text,
                    last_step, size_line, goals_text, stuck_text, view_note)

            # ── API path: two-call Lead+Worker (richer reasoning) ──
            # Phase 1: Lead Agent dictates strategy
            lead_prompt_template = LEAD_AGENT_PROMPT.replace("{role_instructions}", role_instructions)
            lead_full_prompt = (
                f"SYSTEM INSTRUCTIONS:\n{lead_prompt_template}\n\n"
                f"Game Instructions:\n{game_instructions}\n\n"
            )
            if grounding_text:
                lead_full_prompt += f"\n{grounding_text}\n"
            if extra_commands:
                lead_full_prompt += f"\nAdditional available commands:{extra_commands}\n"
            if memories_text:
                lead_full_prompt += f"\n{memories_text}\n"
            if step_history:
                lead_full_prompt += f"\n=== RECENT STEP HISTORY (what you already tried) ===\n{step_history}\n=== END HISTORY ===\n"
                lead_full_prompt += "IMPORTANT: Do NOT repeat actions that already failed or produced no progress. Try a different approach.\n\n"
            if last_step:
                lead_full_prompt += f"\n{last_step}\n"
            if self.turn_count % 5 == 0:
                lead_full_prompt += f"\n\nREMINDER - Core instructions: {game_instructions} | Role: {role_instructions}\n"
            lead_full_prompt += size_line
            lead_full_prompt += "Please analyze the attached screenshot image and provide your strategic directive."

            directive = self._call(lead_full_prompt, image_base64, model_name, expect_json=False)
            print(f"  [lead_agent] Directive: {directive}")

            # Phase 2: Worker Agent generates commands based on directive
            worker_full_prompt = (
                f"SYSTEM INSTRUCTIONS:\n{WORKER_AGENT_PROMPT}\n"
            )
            if extra_commands:
                worker_full_prompt += f"\nAdditional available commands:{extra_commands}\n"
            if last_step:
                worker_full_prompt += f"\n{last_step}\n"
            worker_full_prompt += size_line
            worker_full_prompt += (
                f"\nLead Agent's Directive:\n{directive}\n\n"
                f"Please analyze the attached screenshot image and the Lead Agent's directive to generate the JSON input commands."
            )

            worker_response = self._call(worker_full_prompt, image_base64, model_name, expect_json=True)

            if isinstance(worker_response, dict):
                original_narration = worker_response.get("narration", "")
                worker_response["narration"] = f"Lead Agent Directive: {directive}\n\nWorker Agent Narration: {original_narration}"

            return worker_response
        except BudgetExceededException:
            raise
        except subprocess.TimeoutExpired as e:
            print(f"{self.provider} timeout after {e.timeout}s")
            return {"narration": f"Error: CLI timed out after {e.timeout}s — model may be overloaded, try again", "actions": []}
        except subprocess.CalledProcessError as e:
            stderr_short = (e.stderr or "")[:300]
            print(f"{self.provider} error (exit {e.returncode}): {stderr_short}")
            return {"narration": f"Error: {self.provider} failed (exit {e.returncode}): {stderr_short}", "actions": []}
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {"narration": f"Error: Failed to parse model response as JSON.", "actions": []}
        except Exception as e:
            print(f"Error in LLM integration: {e}")
            return {"narration": f"Error: {e}", "actions": []}
        finally:
            self._extra_image = None

    def _get_next_action_combined(self, image_base64: str, game_instructions: str,
                                   model_name: str, role_instructions: str,
                                   grounding_text: str, extra_commands: str,
                                   step_history: str, memories_text: str,
                                   last_step: str = "", size_line: str = "",
                                   goals_text: str = "", stuck_text: str = "", view_note: str = ""):
        """Single-call path (CLI providers and the Anthropic API): check the
        previous step, plan and generate commands in one model call.

        The prompt is split into a static part (instructions, role, game
        goal, commands) that stays byte-identical between steps — so the
        Anthropic API can cache it — and a dynamic part for this step."""

        prompt_template = COMBINED_AGENT_PROMPT.replace("{role_instructions}", role_instructions)
        static = (
            f"SYSTEM INSTRUCTIONS:\n{prompt_template}\n\n"
            f"Game Instructions:\n{game_instructions}\n\n"
        )
        if extra_commands:
            static += f"\nAdditional available commands:{extra_commands}\n"

        dynamic = ""
        if grounding_text:
            dynamic += f"\n{grounding_text}\n"
        if memories_text:
            dynamic += f"\n{memories_text}\n"
        if step_history:
            dynamic += f"\n=== RECENT STEP HISTORY (what you already tried) ===\n{step_history}\n=== END HISTORY ===\n"
            dynamic += "IMPORTANT: Do NOT repeat actions that already failed or produced no progress. Try a different approach.\n\n"
        if goals_text:
            dynamic += f"\n{goals_text}\n"
        if last_step:
            dynamic += f"\n{last_step}\n"
        if stuck_text:
            dynamic += f"\n{stuck_text}\n"
        if view_note:
            dynamic += f"\n{view_note}\n"
        if self.turn_count % 5 == 0:
            dynamic += f"\n\nREMINDER - Core instructions: {game_instructions} | Role: {role_instructions}\n"
        dynamic += size_line
        dynamic += "Please analyze the attached screenshot image and produce the JSON response."

        if self.provider == ANTHROPIC_PROVIDER:
            response = self._call_anthropic(static, dynamic, image_base64, model_name,
                                            expect_json=True, schema=STEP_SCHEMA)
        elif self.provider in CLI_PROVIDERS:
            response = self._call_cli(static + dynamic, image_base64, model_name, expect_json=True)
        else:
            try:
                response = self._call_api(static + dynamic, image_base64, model_name,
                                          expect_json=True, schema=STEP_SCHEMA)
            except ValueError as e:
                # No parsable JSON (e.g. the model wrote tool-call XML): ask once more.
                print(f"  [api] unparsable step reply ({str(e)[:80]}) — retrying once")
                response = self._call_api(
                    static + dynamic + "\nYour previous reply was not valid JSON. Reply with ONLY the JSON object "
                    "described in FORMAT — no tool calls, no XML, no prose.",
                    image_base64, model_name, expect_json=True, schema=STEP_SCHEMA)
        print(f"  [combined_agent] narration: {response.get('narration', '')[:120]}...")
        return response

    def plan_goals(self, image_base64: str, instructions: str, role: str, model_name: str) -> list:
        """Split the user's instructions into ordered, checkable goals (GCA-010).
        Game agnostic: the goals come only from the instructions and the screenshot."""
        role_instructions = ROLE_PROMPTS.get(role, ROLE_PROMPTS[DEFAULT_ROLE])
        prompt = GOAL_PLAN_PROMPT.format(role_instructions=role_instructions, instructions=instructions)
        try:
            result = self._call(prompt, image_base64, model_name, expect_json=True)
            goals = [str(g).strip() for g in result.get("goals", []) if str(g).strip()]
            return goals[:8]
        except BudgetExceededException:
            raise
        except Exception as e:
            print(f"[goals] planning failed: {e}")
            return []

    def write_report(self, image_base64: str, role: str, instructions: str, session_data: str,
                     model_name: str) -> str:
        """Write the role's end-of-run report from the session's findings (GCA-011)."""
        prompt = REPORT_PROMPT.format(report_kind=REPORT_PROMPTS.get(role, REPORT_PROMPTS["gamer"]),
                                      instructions=instructions, session_data=session_data)
        return self._call(prompt, image_base64, model_name, expect_json=False)

    def retry_assist(self, image_base64: str, failed_cmd: str, failed_reason: str,
                     attempts: int, game_instructions: str, model_name: str):
        """Mid-retry LLM consultation: show the LLM what's failing and get a
        corrected action or 'skip' directive.
        Returns (command_str, reason_str) or ("skip", reason_str) or None on error.
        """
        filled_prompt = RETRY_ASSIST_PROMPT.format(
            failed_action=failed_cmd,
            failed_reason=failed_reason or "(no reason given)",
            attempts=attempts
        )
        prompt = (
            f"SYSTEM INSTRUCTIONS:\n{filled_prompt}\n\n"
            f"Game context:\n{game_instructions}\n\n"
            f"Analyze the current screenshot and provide a corrected action or skip."
        )
        try:
            result = self._call(prompt, image_base64, model_name)
            command = result.get("command", "").strip()
            reason = result.get("reason", "").strip()
            if not command:
                print(f"[retry-assist] LLM returned no command: {result}")
                return None
            return (command, reason)
        except BudgetExceededException as e:
            if hasattr(e, 'partial_result') and isinstance(e.partial_result, dict):
                command = e.partial_result.get("command", "").strip()
                reason = e.partial_result.get("reason", "").strip()
                if command:
                    e.partial_result = (command, reason)
                else:
                    e.partial_result = None
            raise
        except Exception as e:
            print(f"[retry-assist] Error: {e}")
            return None

    def revalidate_actions(self, image_base64: str, remaining_actions: list,
                           game_instructions: str, model_name: str, enabled_tools: list[str] = None):
        """Lightweight LLM call to update coordinates for remaining actions.
        Also handles unexpected prompts/dialogs if the LLM detects one.
        Returns list of action dicts [{"command": ..., "reason": ...}] or None.
        """
        actions_text = "\n".join(
            f"  {i+1}. {cmd}" + (f" — {reason}" if reason else "")
            for i, (cmd, reason) in enumerate(remaining_actions)
        )

        short_cmds = []
        tools_path = os.path.join(os.path.dirname(__file__), 'tools.json')
        if os.path.exists(tools_path):
            with open(tools_path, 'r', encoding='utf-8') as f:
                tools = json.load(f)
            for name, tool in tools.items():
                if enabled_tools is None or name in enabled_tools:
                    short_cmds.append(tool.get('short_description', tool.get('description', '')))

        commands_str_short = "\n".join(short_cmds)
        filled_prompt = REVALIDATE_PROMPT.replace("{commands_list_short}", commands_str_short)

        prompt = (
            f"SYSTEM INSTRUCTIONS:\n{filled_prompt}\n\n"
            f"Game context:\n{game_instructions}\n\n"
            f"REMAINING PLANNED ACTIONS:\n{actions_text}\n\n"
            f"Analyze the current screenshot and provide updated action coordinates."
        )
        try:
            result = self._call(prompt, image_base64, model_name)
            return result.get("actions", [])
        except BudgetExceededException as e:
            if hasattr(e, 'partial_result') and isinstance(e.partial_result, dict):
                e.partial_result = e.partial_result.get("actions", [])
            raise
        except Exception as e:
            print(f"[revalidate] Error: {e}")
            return None

    def verify_action_success(self, after_base64: str, action_taken: str, model_name: str = "gemini-2.5-flash") -> tuple:
        """Lightweight check to see if the action actually changed the game state.
        Note: Passing a single 'after' image relies on the LLM deducing success from the
        current state alone to save context limits and ensure API compatibility.
        Returns (did_succeed: bool, reason: str).
        """
        verify_prompt = """
You are a verification agent checking if a specific action succeeded in a game.

ACTION TAKEN: {action_taken}

Look at the screenshot and determine if this action appears to have been executed.
Be LENIENT — if the screen looks different from what you'd expect BEFORE the action
(e.g. a card is gone from hand, a button state changed, an animation is playing),
assume the action succeeded. Only return false if you are confident the game state
is identical to before the action was attempted.

Respond ONLY with JSON:
{
  "did_succeed": true or false,
  "reason": "Brief explanation."
}
"""
        filled_prompt = verify_prompt.replace("{action_taken}", action_taken)
        prompt = f"SYSTEM: {filled_prompt}\nAnalyze the screenshot and determine if the action appears to have executed successfully."
        try:
            result = self._call(prompt, after_base64, model_name, expect_json=True)
            succeeded = result.get("did_succeed", True)
            reason = result.get("reason", "")
            print(f"  [verify] did_succeed={succeeded}  reason={reason}")
            return succeeded, reason
        except Exception as e:
            print(f"[verify] Verification failed (assuming success to avoid hard blocks): {e}")
            return True, "verification error — defaulting to success"
