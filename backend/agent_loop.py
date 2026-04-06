import asyncio
from typing import Callable, Optional
from datetime import datetime
import time
import ctypes

import pygetwindow as gw

from screen_capture import ScreenCapture
from input_controller import InputController, _send_button, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
from action_verifier import ActionVerifier
from game_state import detect_phase, PHASE_TOOLS
from llm import LLMIntegration, BudgetExceededException
from logger import SessionLogger, ExecutionLogger, ConsoleLogger
from safety import SafetyFilter
from grounding import LLMGrounding
from experience_store import ExperienceStore
from recipes import RecipeStore
from session_state import SessionState
from memories import MemoryStore

# Struggle thresholds — pause agent if actions are taking too many retries
# Average attempts > this means the agent is struggling (1.0 = every action first-try)
STRUGGLE_AVG_THRESHOLD = 4.0
# If ANY single action takes this many attempts, pause regardless of average
STRUGGLE_MAX_THRESHOLD = 12

class AgentLoop:
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.task: Optional[asyncio.Task] = None
        self._resume_event: Optional[asyncio.Event] = None
        
        self.llm = None  # set during start(); exposed for cost tracking
        self.screen_capture = ScreenCapture()
        self.input_controller = InputController()
        self.verifier = ActionVerifier()
        self.safety = SafetyFilter()
        self.experience = ExperienceStore()
        self.recipes = RecipeStore()
        self.session_state = SessionState()
        self.memories = MemoryStore()

    def resume(self):
        """Resume the agent after a pause."""
        if self.is_paused and self._resume_event:
            self.is_paused = False
            self._resume_event.set()
            return True, "Agent resumed."
        return False, "Agent is not paused."

    def abort(self):
        """Abort the agent from a paused state."""
        if self.is_paused and self._resume_event:
            self.is_paused = False
            self.is_running = False
            self._resume_event.set()  # unblock so the loop can exit
            return True, "Agent aborted."
        return self.stop()

    @staticmethod
    def _focus_window(target_type: str, target_name: str):
        """Re-activate the game window so actions land on the right target."""
        if target_type != "window":
            return
        try:
            wins = gw.getWindowsWithTitle(target_name)
            if not wins:
                print(f"  [!] _focus_window: no window found matching '{target_name}'")
                return
            win = wins[0]
            if win.isMinimized:
                win.restore()
                time.sleep(0.1)
            # Try pygetwindow activate first
            try:
                win.activate()
            except Exception:
                pass
            # Alt-key trick: Windows blocks SetForegroundWindow unless the caller
            # is the foreground process. Simulating an Alt press/release tricks
            # Windows into allowing the switch.
            try:
                hwnd = win._hWnd
                ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)  # Alt down
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                ctypes.windll.user32.keybd_event(0x12, 0, 2, 0)  # Alt up
            except Exception:
                pass
            time.sleep(0.2)
            print(f"  [focus] Activated window: '{win.title}'")
        except Exception as e:
            print(f"  [!] _focus_window error: {e}")
        
    def start(self, api_key: str, target_type: str, target_name: str, model_name: str, game_instructions: str, emit_log: Callable,
               provider: str = "gemini_cli", role: str = "gamer", use_grounding: bool = False,
               grounding_model: str = "", max_budget_usd: float = 0.0,
               secondary_api_key: str = ""):
        if self.is_running:
            return False, "Agent is already running."
            
        self.is_running = True
        self.llm = None  # will be set in _loop; exposed for cost tracking
        self.task = asyncio.create_task(self._loop(api_key, target_type, target_name, model_name, game_instructions, emit_log, provider, role, use_grounding, grounding_model, max_budget_usd, secondary_api_key))
        return True, "Agent started."
        
    def stop(self):
        if not self.is_running:
            return False, "Agent is not running."
            
        self.is_running = False
        if self.is_paused and self._resume_event:
            self._resume_event.set()
        if self.task:
            self.task.cancel()
        return True, "Agent stopped."

    @staticmethod
    def _parse_action(action):
        """Normalize action to (command_str, reason_str, precondition, wait_after_condition) regardless of format."""
        if isinstance(action, dict):
            return action.get("command", ""), action.get("reason", ""), action.get("precondition", ""), action.get("wait_after_condition", "")
        # Fallback for plain string actions
        return str(action), "", "", ""

    async def _wait_for_stable(self, target_type: str, target_name: str,
                               timeout: float = 4.0, interval: float = 0.5) -> bool:
        """Poll screen until it stops changing (animation finished).
        Uses perceptual hashing to tolerate idle animations/particles.
        Returns True if screen stabilised, False if timeout reached.
        """
        prev_hash = await asyncio.to_thread(
            self.screen_capture.capture_phash, target_type, target_name)
        elapsed = 0.0
        stable_count = 0
        while elapsed < timeout:
            await asyncio.sleep(interval)
            elapsed += interval
            cur_hash = await asyncio.to_thread(
                self.screen_capture.capture_phash, target_type, target_name)
            if ScreenCapture.hashes_similar(prev_hash, cur_hash):
                stable_count += 1
                if stable_count >= 2:  # stable for 2 consecutive checks
                    return True
            else:
                stable_count = 0
                prev_hash = cur_hash
        return False

    async def _loop(self, api_key: str, target_type: str, target_name: str, model_name: str, game_instructions: str, emit_log: Callable,
                     provider: str = "gemini_cli", role: str = "gamer", use_grounding: bool = False,
                     grounding_model: str = "", max_budget_usd: float = 0.0,
                     secondary_api_key: str = ""):
        # emit_log sends to WebSocket (user-facing)
        # _console_log sends to backend console AND per-session log file
        console_log = ConsoleLogger()
        async def _console_log(msg):
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line)
            console_log.log(line)

        await _console_log("Agent initialized. Starting session logger...")
        
        try:
            llm = LLMIntegration(provider=provider, api_key=api_key if provider != "gemini_cli" else None,
                                 secondary_api_key=secondary_api_key if provider != "gemini_cli" else None,
                                 max_budget_usd=max_budget_usd)
            self.llm = llm  # expose for cost tracking
        except Exception as e:
            await emit_log(f"ERROR: Failed to initialize LLM Integration: {e}")
            self.is_running = False
            return

        # Optional grounding provider for UI element detection
        # Use a separate (potentially cheaper/faster) model for grounding
        _grounding_model = grounding_model or model_name
        grounder = LLMGrounding(llm) if use_grounding else None
        if grounder:
            await _console_log(f"Visual grounding ENABLED — using {_grounding_model} for element detection")
            await emit_log(f"STATUS: Visual grounding enabled (model: {_grounding_model})")
            
        logger = SessionLogger()
        exec_log = ExecutionLogger()
        await _console_log(f"Console log: {console_log.get_filename()}")
        await _console_log(f"Narration log: {logger.get_filename()}")
        await _console_log(f"Execution log: {exec_log.get_filename()}")
        
        step = 1
        consecutive_errors = 0
        STEP_HISTORY_MAX = 5  # keep last N step summaries for LLM context
        step_history = []     # list of short summary strings
        
        async def _handle_budget_call(coro):
            """Helper to run an LLM call and cleanly handle budget exceptions."""
            try:
                return await coro, True
            except BudgetExceededException as e:
                # Capture successful result if available
                result = getattr(e, 'partial_result', None)
                msg_prefix = "after successful API call" if result else "before API call"

                await _console_log(f"  [!] Budget exceeded {msg_prefix}: {e} — pausing agent")
                await emit_log("PAUSED: Budget exceeded. Please authorize an increase in the budget to continue.")

                self.is_paused = True
                self._resume_event = asyncio.Event()
                await self._resume_event.wait()
                self._resume_event = None
                self.is_paused = False

                if not self.is_running:
                    exec_log.log_abort()
                    await _console_log("Agent aborted by user.")
                    return None, False

                # Assume resumed means authorized: double the budget (or add $1 minimum)
                llm.cost.max_budget_usd = max(1.0, llm.cost.max_budget_usd * 2)
                exec_log.log_resume()
                await _console_log(f"Agent resumed by user. New budget: ${llm.cost.max_budget_usd:.2f}")
                await emit_log(f"STATUS: Agent resumed — budget increased to ${llm.cost.max_budget_usd:.2f}, continuing...")

                return result, True

        while self.is_running:
            try:
                exec_log.log_step_header(step, model_name, target_name)
                await _console_log(f"Step {step} - Capturing screen ({target_name})...")
                # Capture screen (includes raw PIL image for template matching)
                capture_result = self.screen_capture.capture(target_type, target_name)
                img_b64 = capture_result["image"]
                reference_pil = capture_result["pil_image"]
                offset_x = capture_result["offset_x"]
                offset_y = capture_result["offset_y"]
                scale_x = capture_result["scale_x"]
                scale_y = capture_result["scale_y"]
                
                # Draw coordinate rulers on the LLM copy (not the template-matching reference)
                ruler_img = self.screen_capture.draw_rulers(reference_pil)
                img_b64_with_rulers = self.screen_capture.pil_to_base64(ruler_img)

                await _console_log(f"Step {step} - Analyzing with {model_name}... offset=({offset_x},{offset_y}) scale=({scale_x:.2f},{scale_y:.2f})")
                await emit_log(f"THINKING: Step {step} — analyzing screen...")
                # ── Experience / Recipe lookup ──
                step_phash = await asyncio.to_thread(
                    self.screen_capture.capture_phash, target_type, target_name)

                # Try recipe replay first — skip LLM entirely if we have a
                # high-confidence recipe for this screen state.
                matched_recipe = self.recipes.match(step_phash)
                if matched_recipe:
                    await _console_log(f"  [recipe] Matched '{matched_recipe.name}' — replaying {len(matched_recipe.actions)} action(s) without LLM")
                    await emit_log(f"RECIPE: Replaying '{matched_recipe.name}' (saved {len(matched_recipe.actions)} actions)")
                    exec_log.log(f"  [recipe] Replaying '{matched_recipe.name}' — {matched_recipe.success_count} prior successes")

                    narration = f"Replaying recipe: {matched_recipe.name}"
                    raw_actions = matched_recipe.actions
                    _active_recipe = matched_recipe
                else:
                    _active_recipe = None

                    experience_text = ""
                    past = self.experience.recall(step_phash, role=role)
                    if past:
                        experience_text = self.experience.format_for_prompt(past)
                        await _console_log(f"  [experience] Recalled {len(past)} similar past experience(s)")

                    # ── Optional grounding pre-pass ──
                    grounding_text = ""
                    if grounder:
                        await _console_log(f"  [grounding] Detecting UI elements...")
                        grounding_result = await asyncio.to_thread(
                            grounder.detect, img_b64_with_rulers, model_name=_grounding_model)
                        if grounding_result.elements:
                            grounding_text = grounding_result.format_for_prompt()
                            await _console_log(f"  [grounding] Found {len(grounding_result.elements)} elements")
                            exec_log.log(f"  [grounding] {len(grounding_result.elements)} UI elements detected")
                        else:
                            await _console_log(f"  [grounding] No elements detected — falling back to rulers only")

                    # Add workflow objective context
                    objective = self.session_state._state.get("objective")
                    if objective:
                        experience_text += f"\n\n=== CURRENT OBJECTIVE / QUEST ===\n{objective}\n================================="

                    # Combine extra context for the LLM
                    extra_context = "\n\n".join(filter(None, [grounding_text, experience_text]))

                    # ── Contextual Tool Pools ──
                    current_phase, _ = await asyncio.to_thread(detect_phase, reference_pil)
                    active_tools = PHASE_TOOLS.get(current_phase, PHASE_TOOLS["unknown"])
                    await _console_log(f"  [tools] Detected phase '{current_phase}' — limiting to {len(active_tools)} tools")

                    # Run the synchronous LLM call in a thread to avoid blocking asyncio loop
                    history_text = "\n".join(step_history) if step_history else ""
                    memories_text = self.memories.format_for_prompt(game=game_instructions[:60])
                    response, should_continue = await _handle_budget_call(
                        asyncio.to_thread(llm.get_next_action, img_b64_with_rulers, game_instructions, model_name, role, extra_context, enabled_tools=active_tools, step_history=history_text, memories_text=memories_text)
                    )
                    if not should_continue:
                        break
                    if response is None:
                        continue  # Retry step
                    
                    narration = response.get("narration", "No narration provided.")
                    raw_actions = response.get("actions", [])

                # Detect LLM error responses (no actions = something went wrong)
                if not raw_actions and ("error" in narration.lower() or "Error" in narration):
                    consecutive_errors += 1
                    cooldown = min(consecutive_errors * 5, 30)  # 5s, 10s, 15s... up to 30s
                    await emit_log(f"ERROR: {narration} (#{consecutive_errors}, cooling down {cooldown}s)")
                    await _console_log(f"  [!] LLM error #{consecutive_errors}: {narration[:200]}, cooldown {cooldown}s")
                    await asyncio.sleep(cooldown)
                    step += 1
                    continue
                consecutive_errors = 0
                
                # Parse actions into (command, reason) tuples
                parsed_actions = [self._parse_action(a) for a in raw_actions]
                
                # Send narration to user-facing telemetry
                await emit_log(f"NARRATION: {narration}")
                for cmd, reason, *_ in parsed_actions:
                    label = f"{cmd}" + (f"  —  {reason}" if reason else "")
                    await emit_log(f"ACTION: {label}")
                
                # Log to markdown
                logger.log_step(step, narration, parsed_actions)
                
                # Snapshot before any actions to detect overall change
                pre_step_pil = await asyncio.to_thread(
                    self.screen_capture.capture_fresh, target_type, target_name)

                # Re-focus game window before executing (focus drifts during LLM call)
                await asyncio.to_thread(self._focus_window, target_type, target_name)

                MAX_RETRIES_CLICK = 5   # 5 nudge attempts; if target isn't found, re-plan
                MAX_RETRIES_DRAG  = 7   # drags are finicky but cap waste at ~30s
                LLM_ASSIST_AFTER  = 3   # ask LLM for help sooner to avoid blind retries
                failed_actions = 0
                total_actions = 0
                attempt_counts = []  # per-action attempt counts for struggle detection

                # Execute actions ONE BY ONE with vision verification + retry
                # Using a while-loop so we can splice in revalidated actions mid-flight
                from action_verifier import MIN_CONFIDENCE as VISION_CONF_THRESHOLD
                action_index = 0

                # Cache the detected phase for the step to avoid re-running OCR repeatedly
                current_phase, _ = await asyncio.to_thread(detect_phase, pre_step_pil)

                while action_index < len(parsed_actions) and self.is_running:
                    cmd, reason, precond, wait_after = parsed_actions[action_index]

                    action_label = f"[{action_index+1}/{len(parsed_actions)}] {cmd}"
                    if reason:
                        action_label += f" ({reason})"
                    await _console_log(f"EXEC: {action_label}")
                    exec_log.log_action_start(action_index + 1, len(parsed_actions), cmd, reason)

                    # ── Precondition check (PR #8: Smart Preconditions) ──
                    if precond:
                        precond_lower = precond.lower()
                        if "==" in precond_lower:
                            req_phase = precond_lower.split("==")[1].strip()
                        elif "=" in precond_lower:
                            req_phase = precond_lower.split("=")[1].strip()
                        else:
                            req_phase = precond_lower.strip()
                        if req_phase and req_phase not in current_phase.lower():
                            msg = f"Skipping action {action_index+1}: precondition '{precond}' failed (current phase: {current_phase})"
                            await _console_log(f"  [!] {msg}")
                            await emit_log(f"WARNING: {msg}")
                            exec_log.log(f"  [skip] {msg}")
                            action_index += 1
                            continue

                    # ── Workflow State (Quest Tracker) ──
                    if cmd.startswith("set_objective"):
                        objective_text = cmd.replace("set_objective", "").strip()
                        if not objective_text and reason:
                            objective_text = reason
                        self.session_state._state["objective"] = objective_text
                        await _console_log(f"  [quest tracker] Objective updated: {objective_text}")
                        await emit_log(f"STATUS: Objective updated to: {objective_text}")
                        action_index += 1
                        continue

                    # ── Save Memory ──
                    if cmd.startswith("save_memory"):
                        memory_text = cmd.replace("save_memory", "").strip()
                        if memory_text:
                            self.memories.add(content=memory_text, game="", tags=["agent-discovered"], source="agent")
                            await _console_log(f"  [memory] Saved: {memory_text}")
                            await emit_log(f"STATUS: Memory saved — {memory_text}")
                        action_index += 1
                        continue

                    # ── Coordinate clamping (fix OOB instead of blocking) ──
                    cmd, was_clamped, clamp_detail = self.safety.clamp_coordinates(cmd)
                    if was_clamped:
                        await _console_log(f"  [safety] Clamped OOB coords: {clamp_detail} -> {cmd}")
                        exec_log.log(f"  [safety] Clamped OOB coords: {clamp_detail} -> {cmd}")
                        # Update the parsed action tuple so downstream code sees clamped cmd
                        parsed_actions[action_index] = (cmd, reason, *parsed_actions[action_index][2:])

                    # ── Safety check ──
                    safe, block_reason = self.safety.check(cmd, reason)
                    if not safe:
                        await _console_log(f"  [safety] BLOCKED: {block_reason}")
                        await emit_log(f"SAFETY: Blocked action {action_index+1} — {block_reason}")
                        exec_log.log(f"  [safety] BLOCKED action {action_index+1}: {block_reason}")
                        total_actions += 1
                        failed_actions += 1
                        attempt_counts.append(1)
                        action_index += 1
                        continue

                    action_succeeded = False
                    final_attempt = 0
                    after_pil = None

                    action_type = cmd.split()[0].lower()
                    is_drag = action_type == "drag"
                    is_keyboard = action_type in ("press", "hold_key", "release_key", "type")
                    is_wait = action_type == "wait"
                    is_macro = action_type == "run_macro"

                    # `wait` commands always succeed — no screen change expected
                    if is_wait:
                        exec_log.log_exec(0, cmd, 0, offset_x, offset_y, scale_x)
                        self.input_controller.execute_action(cmd, offset_x, offset_y, scale_x, scale_y)
                        exec_log.log_result(0, True, True)
                        await _console_log(f"  ✓ Wait completed")
                        total_actions += 1
                        attempt_counts.append(1)
                        exec_log.log_action_outcome(action_index + 1, True, 1)
                        action_index += 1
                        continue

                    # `run_macro` commands bypass vision verification
                    if is_macro:
                        exec_log.log_exec(0, cmd, 0, offset_x, offset_y, scale)
                        # Run the macro and then wait for screen stability
                        await asyncio.to_thread(self.input_controller.execute_action, cmd, offset_x, offset_y, scale)
                        stable = await self._wait_for_stable(target_type, target_name, timeout=4.0, interval=0.4)
                        exec_log.log_result(0, True, stable)
                        await _console_log(f"  ✓ Macro {cmd} completed")
                        total_actions += 1
                        attempt_counts.append(1)
                        exec_log.log_action_outcome(action_index + 1, True, 1)
                        action_index += 1
                        overall_changed = True # Assume macro did something
                        # Update after_pil so that following actions can be revalidated
                        after_pil = await asyncio.to_thread(self.screen_capture.capture_fresh, target_type, target_name)
                        continue

                    # ── Fast-path: keyboard actions ──
                    # Keyboard presses have no coordinates to verify or nudge.
                    # Execute immediately and move on — like the Cascade model:
                    # do the action, check results at the end of the batch.
                    if is_keyboard:
                        exec_log.log_exec(0, cmd, 0, offset_x, offset_y, scale_x)
                        await _console_log(f"  [exec] {cmd}" + (f"  —  {reason}" if reason else ""))
                        await emit_log({"type": "executing", "command": cmd, "reason": reason, "attempt": 0, "max_retries": 0, "message": f"{cmd}" + (f"  —  {reason}" if reason else "")})
                        self.input_controller.execute_action(cmd, offset_x, offset_y, scale_x, scale_y)
                        await asyncio.sleep(0.3)  # brief pause for game to register
                        exec_log.log_result(0, True, True)
                        await _console_log(f"  ✓ Key action completed")
                        total_actions += 1
                        attempt_counts.append(1)
                        exec_log.log_action_outcome(action_index + 1, True, 1)
                        action_index += 1
                        continue

                    if is_drag:
                        max_retries = MAX_RETRIES_DRAG
                    else:
                        max_retries = MAX_RETRIES_CLICK

                    action_approved = False
                    for attempt in range(max_retries + 1):
                        if not self.is_running:
                            break

                        # ── Vision verification ──
                        fresh_pil = await asyncio.to_thread(
                            self.screen_capture.capture_fresh, target_type, target_name)
                        adjusted_cmd, adjustments = await asyncio.to_thread(
                            self.verifier.verify_and_adjust, cmd, reference_pil, fresh_pil, reason)

                        # On retries, also apply a coordinate nudge to the target
                        if attempt > 0:
                            adjusted_cmd = ActionVerifier.nudge_action(
                                adjusted_cmd, attempt - 1)

                        if adjustments:
                            adj_log = self.verifier.format_adjustment_log(adjustments)
                            await _console_log(f"  [vision] attempt {attempt}\n{adj_log}")
                            exec_log.log_vision(attempt, adj_log)

                        # ── High-risk classification ──
                        if not action_approved:
                            is_risky, keyword = await asyncio.to_thread(
                                self.verifier.check_action_risk, adjusted_cmd, fresh_pil)
                            if is_risky:
                                await emit_log(f"REQUIRES_APPROVAL: High-risk keyword '{keyword}' detected. Please approve to execute: {adjusted_cmd}")
                                await _console_log(f"  [!] High-risk keyword '{keyword}' detected. Pausing for approval.")
                                self.is_paused = True
                                self._resume_event = asyncio.Event()
                                await self._resume_event.wait()
                                self._resume_event = None
                                if not self.is_running:
                                    break
                                action_approved = True
                                await _console_log("  [+] Action approved by user.")
                                await emit_log("STATUS: Action approved — continuing...")

                        attempt_label = f" (retry {attempt})" if attempt > 0 else ""
                        exec_label = adjusted_cmd if adjusted_cmd != cmd else cmd
                        # Only emit to frontend on key attempts to avoid sidebar spam
                        emit_to_frontend = attempt == 0 or attempt % 5 == 0
                        await _console_log(f"  [exec{attempt_label}] {exec_label}")
                        if emit_to_frontend:
                            if adjusted_cmd != cmd or attempt > 0:
                                await emit_log({"type": "executing", "command": exec_label, "reason": reason, "attempt": attempt, "max_retries": max_retries, "message": f"{exec_label}" + (f"  —  {reason}{attempt_label}" if reason else attempt_label)})
                            else:
                                await emit_log({"type": "executing", "command": cmd, "reason": reason, "attempt": attempt, "max_retries": max_retries, "message": f"{cmd}" + (f"  —  {reason}" if reason else "")})

                        # Use the fresh screenshot as "before" for pixel comparison
                        before_pil = fresh_pil

                        # Update reference for next action's template matching
                        reference_pil = fresh_pil

                        # Escalating settle: 0.5s base, +0.05s per attempt, cap 1.5s
                        settle = min(0.5 + attempt * 0.05, 1.5)
                        exec_log.log_exec(attempt, adjusted_cmd, settle,
                                          offset_x, offset_y, scale_x)
                        self.input_controller.execute_action(
                            adjusted_cmd, offset_x, offset_y, scale_x, scale_y, settle_s=settle)

                        # Wait for game to process
                        await asyncio.sleep(0.4)
                        stable = await self._wait_for_stable(target_type, target_name,
                                                              timeout=2.0, interval=0.3)

                        # Check if screen changed (pixel-level comparison, much
                        # more sensitive than the old 16x16 perceptual hash)
                        # Click actions use a slightly higher threshold to filter
                        # idle animations, but not so high that subtle changes
                        # (e.g. playing a card, selecting a map node) are missed.
                        after_pil = await asyncio.to_thread(
                            self.screen_capture.capture_fresh, target_type, target_name)
                        is_click_action = cmd.split()[0].lower().endswith("click")
                        diff_threshold = 0.012 if is_click_action else 0.008
                        changed = ScreenCapture.pil_images_different(
                            before_pil, after_pil, fraction_threshold=diff_threshold)

                        # Drag persistence check: snap-back detection.
                        # Drags can register as CHANGED (pickup animation) even
                        # when the item wasn't actually placed and snapped back.
                        # Wait a bit longer and re-compare to the BEFORE screenshot.
                        if changed and is_drag:
                            await asyncio.sleep(0.6)
                            persist_pil = await asyncio.to_thread(
                                self.screen_capture.capture_fresh, target_type, target_name)
                            still_changed = ScreenCapture.pil_images_different(
                                before_pil, persist_pil, fraction_threshold=0.008)
                            if not still_changed:
                                changed = False
                                exec_log.log(f"  [result] attempt={attempt}  screen=SNAP_BACK (drag reverted)")
                                await _console_log(f"  [!] Drag appeared to succeed but screen reverted (snap-back)")

                        exec_log.log_result(attempt, changed, stable)

                        if changed:
                            # ── Verify Agent Sub-role ──
                            # Only verify on first 2 attempts — after that, trust pixel diff.
                            # Each verify is an extra LLM round-trip; on retries the screen
                            # is more likely to show transient animations that confuse the verifier.
                            VERIFY_MAX_ATTEMPT = 1
                            if (role == "gamer" and getattr(self, 'llm', None)
                                    and attempt <= VERIFY_MAX_ATTEMPT):
                                await _console_log("  [verify] Asking Verify Agent if game state progressed...")
                                after_b64 = self.screen_capture.pil_to_base64(after_pil)
                                verified, verify_reason = await asyncio.to_thread(
                                    self.llm.verify_action_success, after_b64, exec_label, model_name)
                                if not verified:
                                    changed = False
                                    await _console_log(f"  [!] Screen changed visually, but Verify Agent ruled action failed. Reason: {verify_reason}")
                                    exec_log.log(f"  [verify] Failed: {verify_reason}")

                        if changed:
                            await _console_log(f"  ✓ Action {action_index+1} succeeded on attempt {attempt}")
                            action_succeeded = True
                            final_attempt = attempt
                            break
                        else:
                            await _console_log(f"  ✗ Attempt {attempt} — screen unchanged")
                            if attempt < max_retries:
                                # ── Mid-retry LLM consultation ──
                                # After LLM_ASSIST_AFTER failed attempts, ask the LLM
                                # what's wrong instead of continuing blind nudges.
                                if attempt + 1 == LLM_ASSIST_AFTER:
                                    await _console_log(f"  [retry-assist] Asking LLM for help after {attempt+1} failures...")
                                    await emit_log({"type": "llm_assist", "message": f"Asking LLM for help — action {action_index+1} failed {attempt+1} times"})
                                    assist_pil = await asyncio.to_thread(
                                        self.screen_capture.capture_fresh, target_type, target_name)
                                    assist_b64 = ScreenCapture.pil_to_base64(ScreenCapture.draw_rulers(assist_pil))
                                    assist_result, should_continue = await _handle_budget_call(
                                        asyncio.to_thread(llm.retry_assist, assist_b64, cmd, reason,
                                                          attempt + 1, game_instructions, model_name)
                                    )
                                    if not should_continue:
                                        break
                                    if assist_result is None:
                                        continue  # Retry this attempt iteration

                                    if assist_result:
                                        new_cmd, new_reason = assist_result
                                        exec_log.log_retry_assist(attempt + 1, cmd, new_cmd, new_reason)
                                        if new_cmd.lower() == "skip":
                                            await _console_log(f"  [retry-assist] LLM says SKIP: {new_reason}")
                                            await emit_log({"type": "llm", "message": f"skipping action — {new_reason}"})
                                            break  # exit retry loop, mark as failed
                                        else:
                                            await _console_log(f"  [retry-assist] LLM corrected: {new_cmd}")
                                            await emit_log({"type": "llm", "message": f"corrected action: {new_cmd}"})
                                            cmd = new_cmd
                                            reason = new_reason or reason
                                            # Update in parsed_actions so downstream
                                            # revalidation sees the corrected action
                                            parsed_actions[action_index] = (cmd, reason, precond, wait_after)
                                            # Recompute max_retries for new action type
                                            action_type = cmd.split()[0].lower()
                                            is_drag = action_type == "drag"
                                            if is_drag:
                                                max_retries = MAX_RETRIES_DRAG
                                            else:
                                                max_retries = MAX_RETRIES_CLICK
                                            # Re-focus and continue with next attempt
                                            await asyncio.to_thread(self._focus_window, target_type, target_name)
                                            continue  # skip the normal retry logic below

                                # Right-click cancel ONLY for drag actions — cancels
                                # an item stuck to cursor after a failed drag.  For click
                                # actions this is harmful: right-clicking on a button
                                # opens tooltips/overlays that block subsequent clicks.
                                is_drag = cmd.split()[0].lower() == "drag"
                                if is_drag:
                                    _send_button(MOUSEEVENTF_RIGHTDOWN)
                                    await asyncio.sleep(0.05)
                                    _send_button(MOUSEEVENTF_RIGHTUP)
                                    await asyncio.sleep(0.3)
                                # Re-focus before retry
                                await asyncio.to_thread(self._focus_window, target_type, target_name)
                                await _console_log(f"  ↻ Retrying with nudge + longer settle...")
                                if (attempt + 1) % 5 == 0:
                                    await emit_log({"type": "action_retry", "action_index": action_index + 1, "attempt": attempt + 1, "max_retries": max_retries, "message": f"Retrying action {action_index+1} (attempt {attempt+1}/{max_retries})"})

                    total_actions += 1
                    if not action_succeeded:
                        failed_actions += 1
                        final_attempt = max_retries
                        await _console_log(f"  ✗✗ Action {action_index+1} FAILED after {max_retries} retries")
                        await emit_log({"type": "warning", "message": f"Action failed after {max_retries} retries — skipping"})
                    attempt_counts.append(final_attempt + 1)
                    exec_log.log_action_outcome(action_index + 1, action_succeeded, final_attempt + 1)

                    if not stable:
                        await _console_log(f"  … Screen still animating (timeout), proceeding")

                    # ── wait_after condition (PR #8: Smart Preconditions) ──
                    if wait_after and self.is_running:
                        await _console_log(f"  [wait_after] Waiting for condition: '{wait_after}' to resolve...")
                        await emit_log(f"WAITING: Waiting for {wait_after}...")
                        waited_stable = await self._wait_for_stable(target_type, target_name, timeout=10.0, interval=0.5)
                        if waited_stable:
                            await _console_log(f"  [wait_after] Screen stabilized.")
                        else:
                            await _console_log(f"  [wait_after] Timeout waiting for screen to stabilize.")

                    # ── Proactive re-planning ──
                    REPLAN_THRESHOLD = 0.15  # 15% pixel change = major UI shift
                    if action_succeeded and after_pil is not None and action_index < len(parsed_actions) - 1:
                        major_change = ScreenCapture.pil_images_different(
                            pre_step_pil, after_pil, fraction_threshold=REPLAN_THRESHOLD)

                        if major_change:
                            remaining_count = len(parsed_actions) - action_index - 1
                            await _console_log(f"  [replan] Major screen change detected — discarding {remaining_count} remaining action(s)")
                            await emit_log(f"RE-PLANNING: Screen changed significantly — getting fresh analysis")
                            exec_log.log(f"  [replan] Major change after action {action_index+1}, discarding {remaining_count} remaining action(s)")
                            action_index += 1
                            break

                        # Minor change — check if next action's coordinates still valid
                        next_cmd = parsed_actions[action_index + 1][0]
                        conf = await asyncio.to_thread(
                            self.verifier.check_action_confidence,
                            next_cmd, reference_pil, after_pil)

                        if conf < VISION_CONF_THRESHOLD:
                            await _console_log(f"  [revalidate] Next action confidence {conf:.2f} < {VISION_CONF_THRESHOLD} — requesting LLM update...")
                            await emit_log(f"REVALIDATING: Updating remaining action coordinates...")

                            # revalidate_actions expects a list of (cmd, reason) tuples
                            remaining = [(cmd, reason) for cmd, reason, _, _ in parsed_actions[action_index + 1:]]
                            img_b64 = ScreenCapture.pil_to_base64(ScreenCapture.draw_rulers(after_pil))
                            raw_updated, should_continue = await _handle_budget_call(
                                asyncio.to_thread(llm.revalidate_actions, img_b64, remaining,
                                                  game_instructions, model_name)
                            )
                            if not should_continue:
                                break
                            if raw_updated is None:
                                continue  # Retry verification/revalidation

                            if raw_updated:
                                updated = [self._parse_action(a) for a in raw_updated]
                                parsed_actions = parsed_actions[:action_index + 1] + updated
                                reference_pil = after_pil

                                exec_log.log_revalidation(conf, len(updated), [(c, r) for c, r, _, _ in updated])
                                await _console_log(f"  [revalidate] Replaced {len(remaining)} remaining action(s) with {len(updated)} updated action(s)")
                                for j, (ucmd, ureason, _, _) in enumerate(updated):
                                    await _console_log(f"    [{action_index+2+j}] {ucmd}" + (f" — {ureason}" if ureason else ""))
                            else:
                                exec_log.log(f"  [revalidate] LLM call failed — continuing with original coords")
                                await _console_log(f"  [revalidate] LLM update failed, continuing with original coords")
                        else:
                            exec_log.log_revalidation_skip(conf)
                            await _console_log(f"  [revalidate] Next action confidence {conf:.2f} — coords OK")

                    # Re-focus between actions in case focus drifted
                    if action_index < len(parsed_actions) - 1:
                        await asyncio.to_thread(self._focus_window, target_type, target_name)

                    action_index += 1

                # Post-action verification: did anything change at all?
                post_step_pil = await asyncio.to_thread(
                    self.screen_capture.capture_fresh, target_type, target_name)
                overall_changed = ScreenCapture.pil_images_different(pre_step_pil, post_step_pil)

                # ── Same-state retry ──
                # If the step accomplished nothing, retry the last action locally
                # instead of wasting ~40s on another LLM round-trip.  Handles
                # the common case of End Turn not registering due to animation
                # timing or the button not being active yet.
                SAME_STATE_RETRIES = 5
                if not overall_changed and self.is_running and parsed_actions:
                    last_cmd = parsed_actions[-1][0]
                    await _console_log(f"  [same-state] Step {step} had no effect — retrying last action: {last_cmd}")
                    await emit_log(f"RETRYING: {last_cmd} (step had no visible effect)")
                    exec_log.log(f"  [same-state] overall_screen_change=no → retrying last action: {last_cmd}")

                    for ss_retry in range(SAME_STATE_RETRIES):
                        if not self.is_running:
                            break
                        # Wait longer for animations/transitions to finish
                        await asyncio.sleep(1.5)
                        await asyncio.to_thread(self._focus_window, target_type, target_name)

                        self.input_controller.execute_action(
                            last_cmd, offset_x, offset_y, scale_x, scale_y)
                        await asyncio.sleep(0.8)
                        await self._wait_for_stable(target_type, target_name,
                                                    timeout=2.0, interval=0.3)

                        retry_pil = await asyncio.to_thread(
                            self.screen_capture.capture_fresh, target_type, target_name)
                        if ScreenCapture.pil_images_different(pre_step_pil, retry_pil):
                            overall_changed = True
                            await _console_log(f"  [same-state] Retry {ss_retry+1}/{SAME_STATE_RETRIES} succeeded")
                            exec_log.log(f"  [same-state] retry {ss_retry+1} OK — screen changed")
                            break
                        else:
                            await _console_log(f"  [same-state] Retry {ss_retry+1}/{SAME_STATE_RETRIES} — still unchanged")
                            exec_log.log(f"  [same-state] retry {ss_retry+1} — still unchanged")

                if not overall_changed:
                    await _console_log(f"  [!] Step {step}: Screen identical before/after ALL actions — nothing worked")
                    await emit_log(f"WARNING: No actions appeared to take effect this step")
                else:
                    await _console_log(f"  Step {step}: Screen changed — actions had effect")

                # ── Experience recording ──
                # Record what happened this step so we can recall it in future sessions
                step_succeeded = overall_changed and failed_actions == 0
                if raw_actions and step_phash:
                    self.experience.record(
                        phash=step_phash,
                        ocr_snippet=narration[:200],
                        actions=raw_actions,
                        succeeded=step_succeeded,
                        role=role,
                    )

                # ── Recipe tracking ──
                if _active_recipe:
                    # Update the replayed recipe's track record
                    self.recipes.record_result(_active_recipe, step_succeeded)
                    if step_succeeded:
                        await _console_log(f"  [recipe] '{_active_recipe.name}' replay succeeded (total: {_active_recipe.success_count})")
                    else:
                        await _console_log(f"  [recipe] '{_active_recipe.name}' replay FAILED (fails: {_active_recipe.fail_count})")
                elif step_succeeded and raw_actions and step_phash and len(raw_actions) >= 2:
                    # Auto-promote: if the same pattern succeeds 3+ times, create a recipe
                    action_label = raw_actions[0].get("reason", raw_actions[0].get("command", "unnamed"))
                    self.recipes.create_from_experience(
                        name=action_label[:60],
                        phash=step_phash,
                        actions=raw_actions,
                        min_occurrences=3,
                        experience_store=self.experience,
                    )

                # Compute struggle metrics from attempt counts
                avg_attempts = sum(attempt_counts) / len(attempt_counts) if attempt_counts else 1.0
                max_attempts = max(attempt_counts) if attempt_counts else 1
                exec_log.log_step_summary(
                    step, total_actions, failed_actions, overall_changed,
                    avg_attempts, max_attempts)

                # ── Struggle check: pause if actions are taking too many retries ──
                if total_actions > 0:
                    should_pause = (
                        avg_attempts > STRUGGLE_AVG_THRESHOLD
                        or max_attempts >= STRUGGLE_MAX_THRESHOLD
                        or failed_actions > 0
                    )
                    await _console_log(
                        f"  Step {step}: {total_actions-failed_actions}/{total_actions} succeeded, "
                        f"avg_attempts={avg_attempts:.1f}, max={max_attempts}, failed={failed_actions}")
                    if should_pause:
                        reason_parts = []
                        if avg_attempts > STRUGGLE_AVG_THRESHOLD:
                            reason_parts.append(f"avg attempts {avg_attempts:.1f} > {STRUGGLE_AVG_THRESHOLD}")
                        if max_attempts >= STRUGGLE_MAX_THRESHOLD:
                            reason_parts.append(f"max attempts {max_attempts} >= {STRUGGLE_MAX_THRESHOLD}")
                        if failed_actions > 0:
                            reason_parts.append(f"{failed_actions} action(s) failed entirely")
                        pause_reason = "; ".join(reason_parts)

                        exec_log.log_pause_struggle(avg_attempts, max_attempts, failed_actions)
                        await _console_log(f"  [!] Struggling: {pause_reason} — pausing agent")
                        await emit_log(f"PAUSED: Agent is struggling ({pause_reason}). Press Continue Agent or Abort to proceed.")
                        self.is_paused = True
                        self._resume_event = asyncio.Event()
                        await self._resume_event.wait()
                        self._resume_event = None
                        if not self.is_running:
                            exec_log.log_abort()
                            await _console_log("Agent aborted by user.")
                            break
                        exec_log.log_resume()
                        await _console_log("Agent resumed by user.")
                        await emit_log("STATUS: Agent resumed — continuing...")

                # ── Step history for LLM context ──
                action_summaries = []
                for i, (c, r, *_rest) in enumerate(parsed_actions):
                    status = "OK" if (i < len(attempt_counts) and attempt_counts[i] == 1) else "FAILED/RETRIED"
                    action_summaries.append(f"  {c}" + (f" ({r})" if r else "") + f" → {status}")
                outcome = "screen changed" if overall_changed else "NO EFFECT"
                # Include first line of Lead Agent directive (usually has game ID)
                directive_summary = narration.split('\n')[0][:120] if narration else ""
                history_entry = (
                    f"Step {step}: {outcome} | {directive_summary}\n"
                    + "\n".join(action_summaries)
                )
                step_history.append(history_entry)
                if len(step_history) > STEP_HISTORY_MAX:
                    step_history = step_history[-STEP_HISTORY_MAX:]

                # ── Session state checkpoint ──
                self.session_state.save(
                    step=step,
                    target_type=target_type,
                    target_name=target_name,
                    model_name=model_name,
                    game_instructions=game_instructions,
                    provider=provider,
                    role=role,
                    use_grounding=use_grounding,
                    grounding_model=grounding_model,
                    last_narration=narration,
                    last_actions=raw_actions,
                )

                step += 1
                
                # Cooldown to respect rate limits
                await _console_log("Waiting before next cycle...")
                for _ in range(3):
                    if not self.is_running:
                        break
                    await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                await _console_log("Agent task cancelled.")
                break
            except Exception as e:
                await emit_log(f"ERROR: {e}")
                await _console_log(f"Exception in agent loop: {e}")
                await asyncio.sleep(5)
                
        self.session_state.clear()
        self.is_running = False
        await _console_log("Agent loop terminated.")


