import re
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
        "- Still play to make progress — a reviewer needs to experience the game, not stall."
    ),
    "tester": (
        "ROLE: QA Tester — You are testing this game for bugs, glitches, and edge cases.\n"
        "- Systematically explore the UI: try hovering elements, clicking unexpected areas, using wrong inputs.\n"
        "- Attempt actions that might break things: spam clicks, interact during animations, use items in odd contexts.\n"
        "- In your narration, log anything unusual: visual glitches, misaligned elements, unresponsive buttons, text overflow.\n"
        "- Note the reproduction steps for any issue you find.\n"
        "- Alternate between normal gameplay and deliberate edge-case probing.\n"
        "- If something looks like a bug, try to reproduce it before moving on."
    ),
    "speedrunner": (
        "ROLE: Speedrunner — You are trying to complete the game as fast as possible.\n"
        "- Minimize time and unnecessary actions. Skip optional content, dialogue, and animations.\n"
        "- Prefer keyboard shortcuts over mouse clicks for speed.\n"
        "- Take the most efficient path — avoid exploration that doesn't advance the objective.\n"
        "- In your narration, briefly note your routing decisions and time-saving choices.\n"
        "- Accept higher risk if it saves significant time."
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
- press key / hold_key / release_key / type
- click / right_click / double_click / drag / scroll / hover
- wait seconds
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
}
DEFAULT_PRICING = (1.00, 3.00)  # conservative fallback


class BudgetExceededException(Exception):
    """Raised when the session budget limit is reached."""
    pass


class CostTracker:
    """Tracks token usage and estimated cost for API-based providers."""

    def __init__(self, max_budget_usd: float = 0.0):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.max_budget_usd = max_budget_usd

    def record(self, input_tokens: int, output_tokens: int, model_name: str):
        # Prevent API calls if we are already at or over budget
        if self.max_budget_usd > 0.0 and self.total_cost_usd >= self.max_budget_usd:
            raise BudgetExceededException(f"Budget exceeded: {self.total_cost_usd:.4f} >= {self.max_budget_usd:.4f}")

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1
        pricing = MODEL_PRICING.get(model_name, DEFAULT_PRICING)
        cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000
        self.total_cost_usd += cost

        # After updating, check if we crossed the limit
        if self.max_budget_usd > 0.0 and self.total_cost_usd > self.max_budget_usd:
            raise BudgetExceededException(f"Budget exceeded: {self.total_cost_usd:.4f} > {self.max_budget_usd:.4f}")

    def get_summary(self) -> dict:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "call_count": self.call_count,
        }


class LLMIntegration:
    PROVIDER_ENDPOINTS = {
        "gemini_api": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }

    def __init__(self, provider: str = "gemini_cli", api_key: str = None, max_budget_usd: float = 0.0):
        self.provider = provider
        self.api_key = api_key
        self.cost = CostTracker(max_budget_usd)
        self.turn_count = 0
        if provider != "gemini_cli" and _requests is None:
            raise ImportError("'requests' package is required for API providers. "
                              "Install with: pip install requests")

    # ── Routing ──

    def _call(self, prompt_text: str, image_base64: str, model_name: str):
        """Route to the configured provider."""
        if self.provider == "gemini_cli":
            return self._call_gemini_cli(prompt_text, image_base64, model_name)
        else:
            return self._call_api(prompt_text, image_base64, model_name)

    # ── Gemini CLI (free, default) ──

    def _call_gemini_cli(self, prompt_text: str, image_base64: str, model_name: str):
        """Call Gemini via the local CLI tool. Free but slower."""
        if not re.match(r'^[\w\-\.]+$', model_name):
            raise ValueError(f"Invalid model name format: {model_name}")

        temp_img_path = None
        temp_prompt_path = None
        try:
            tmp_dir = os.path.join(os.path.dirname(__file__), "..", ".tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            img_data = base64.b64decode(image_base64)
            fd, temp_img_path = tempfile.mkstemp(suffix=".jpg", dir=tmp_dir)
            with os.fdopen(fd, 'wb') as f:
                f.write(img_data)

            fd2, temp_prompt_path = tempfile.mkstemp(suffix=".txt", dir=tmp_dir)
            with os.fdopen(fd2, 'w', encoding='utf-8') as f:
                f.write(prompt_text)

            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            rel_img_path = os.path.relpath(temp_img_path, workspace_root)
            rel_prompt_path = os.path.relpath(temp_prompt_path, workspace_root)

            full_prompt = f"Analyze this image: @{rel_img_path}. Task instructions: @{rel_prompt_path}"

            result = subprocess.run(
                ["gemini", "-p", full_prompt, "--model", model_name],
                input="1\n",
                cwd=workspace_root,
                capture_output=True, text=True, check=True,
                shell=True
            )
            content = result.stdout.strip()

            # Clean up potential markdown wrappers
            clean_content = re.sub(r'```(?:json)?', '', content).strip()
            # Find the first { and the last }
            start_idx = clean_content.find('{')
            end_idx = clean_content.rfind('}')
            if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
                raise ValueError(f"No JSON object found in output. Raw output: {content}")

            json_str = clean_content[start_idx:end_idx + 1]
            return json.loads(json_str)
        finally:
            for path in [temp_img_path, temp_prompt_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

    # ── API providers (Gemini API / OpenRouter) ──

    def _call_api(self, prompt_text: str, image_base64: str, model_name: str):
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
                    }}
                ]
            }],
            "max_tokens": 2048,
            "temperature": 0.7,
        }

        # Retry transient errors with exponential backoff
        RETRYABLE = {429, 500, 502, 503, 504}
        MAX_API_RETRIES = 3
        last_err = None
        data = None
        elapsed = 0

        for attempt in range(MAX_API_RETRIES + 1):
            try:
                t0 = time.time()
                resp = _requests.post(endpoint, headers=headers, json=body, timeout=120)
                elapsed = time.time() - t0
                resp.raise_for_status()
                data = resp.json()
                break
            except _requests.exceptions.HTTPError as e:
                last_err = e
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

        # Track tokens / cost
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)

        # Parse JSON from content first so we don't lose the result if budget is exceeded
        clean_content = re.sub(r'```(?:json)?', '', content).strip()
        start_idx = clean_content.find('{')
        end_idx = clean_content.rfind('}')
        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            raise ValueError(f"No JSON in API response: {content[:300]}")

        json_str = clean_content[start_idx:end_idx + 1]
        result_json = json.loads(json_str)

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
                        role: str = "gamer", grounding_text: str = "", enabled_tools: list = None):
        try:
            self.turn_count += 1
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
            if self.turn_count % 5 == 0:
                lead_full_prompt += f"\n\nREMINDER - Core instructions: {game_instructions} | Role: {role_instructions}\n"
            lead_full_prompt += "Please analyze the attached screenshot image and provide your strategic directive."

            directive = self._call(lead_full_prompt, image_base64, model_name, expect_json=False)
            print(f"  [lead_agent] Directive: {directive}")

            # Phase 2: Worker Agent generates commands based on directive
            worker_full_prompt = (
                f"SYSTEM INSTRUCTIONS:\n{WORKER_AGENT_PROMPT}\n"
            )
            if extra_commands:
                worker_full_prompt += f"\nAdditional available commands:{extra_commands}\n"
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
        except subprocess.CalledProcessError as e:
            print(f"Gemini CLI error: {e.stderr}")
            return {"narration": f"CLI Error occurred: {e.stderr}", "actions": []}
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {"narration": f"Failed to parse model response as JSON.", "actions": []}
        except Exception as e:
            print(f"Error in LLM integration: {e}")
            return {"narration": f"Error occurred: {e}", "actions": []}

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
