import asyncio
import re
from typing import Callable, Optional
from datetime import datetime
import time
import ctypes

from screen_capture import ScreenCapture, WindowNotFoundError, find_window, is_streaming_client, close_window
from input_controller import InputController, set_target_window, set_streaming
from text_locator import find_text
from action_verifier import ActionVerifier
from game_state import detect_phase, PHASE_TOOLS
from llm import LLMIntegration, BudgetExceededException, is_cli_provider, escalation_model
from logger import SessionLogger, ExecutionLogger, ConsoleLogger
from safety import SafetyFilter
from grounding import LLMGrounding
from experience_store import ExperienceStore
from recipes import RecipeStore
from session_state import SessionState
from memories import MemoryStore
from goals import GoalTracker
from notebook import RunNotebook
from reporting import FindingsLog

# Progress, escalation and skipping are handled by GoalTracker (GCA-009/010).
# Single "failed" actions never pause: the model judges each step itself.

# Model-requested waits are capped (GCA-013): the harness already settles.
MAX_MODEL_WAIT_S = 2.0

# Zoom is for small targets; wider boxes are refused (GCA-014).
MAX_ZOOM_BOX_W = 480

# Keep findings notable: the model tends to log everything it sees.
MAX_FINDINGS_PER_STEP = 2

# Pixel-diff threshold used only as a hint to the model ("screen changed: yes/no").
CHANGE_HINT_THRESHOLD = 0.004

# Before each planning screenshot, wait up to this long for the screen to settle (GCA-007).
SETTLE_TIMEOUT_S = 3.0

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
        self._timer_task: Optional[asyncio.Task] = None
        self.unattended = False  # timed run: refuse risky clicks instead of waiting for approval
        self.dual_frame = False  # send a second frame ~0.4 s later so animations are visible (GCA-025)
        self.timer_state: dict = {}  # {"stop_at": epoch} or {"close_at": epoch, "close_target": title}

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
            win = find_window(target_name)
            set_target_window(win._hWnd)
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

    @staticmethod
    def _memory_game(target_type: str, target_name: str) -> str:
        """Game name memories are filed under: the window title without
        build tags such as '(DEBUG)'. Monitor targets get universal memories."""
        if target_type != "window":
            return ""
        return re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", target_name).strip()

    def start(self, api_key: str, target_type: str, target_name: str, model_name: str, game_instructions: str, emit_log: Callable,
               provider: str = "gemini_cli", role: str = "gamer", use_grounding: bool = False,
               grounding_model: str = "", max_budget_usd: float = 0.0,
               secondary_api_key: str = "", capture_mode: str = "auto",
               max_minutes: float = 0.0, close_target_after_minutes: float = -1.0,
               dual_frame: bool = False):
        if self.is_running:
            return False, "Agent is already running."

        self.is_running = True
        self.llm = None  # will be set in _loop; exposed for cost tracking
        self.dual_frame = bool(dual_frame)
        self.task = asyncio.create_task(self._loop(api_key, target_type, target_name, model_name, game_instructions, emit_log, provider, role, use_grounding, grounding_model, max_budget_usd, secondary_api_key, capture_mode))
        # A new run replaces any pending time limit / close from an earlier run.
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self.timer_state = {}
        self.unattended = bool(max_minutes and max_minutes > 0)
        if max_minutes and max_minutes > 0:
            self._timer_task = asyncio.create_task(self._run_time_limit(
                max_minutes, close_target_after_minutes, target_type, target_name, emit_log))
        return True, "Agent started."

    async def _run_time_limit(self, max_minutes: float, close_after_minutes: float,
                              target_type: str, target_name: str, emit_log: Callable):
        """Stop the run after `max_minutes` (the report is still written), then
        optionally close the game window `close_after_minutes` later — e.g. to
        end a GeForce NOW session instead of leaving it idling."""
        stop_at = time.time() + max_minutes * 60
        self.timer_state = {"stop_at": stop_at}
        while time.time() < stop_at:
            if not self.is_running:
                self.timer_state = {}
                return  # stopped by hand: no automatic close
            await asyncio.sleep(1.0)
        if not self.is_running:
            self.timer_state = {}
            return
        print(f"[timer] {max_minutes:g}-minute limit reached — stopping the agent")
        await emit_log(f"STATUS: Time limit of {max_minutes:g} minutes reached — stopping the test")
        self.stop()
        if close_after_minutes is None or close_after_minutes < 0 or target_type != "window":
            self.timer_state = {}
            return
        close_at = time.time() + close_after_minutes * 60
        self.timer_state = {"close_at": close_at, "close_target": target_name}
        await emit_log(f"STATUS: Will close '{target_name}' in {close_after_minutes:g} minutes")
        await asyncio.sleep(max(0.0, close_at - time.time()))
        if self.is_running:  # a new run started on its own: leave the window alone
            self.timer_state = {}
            return
        result = await asyncio.to_thread(close_window, target_name)
        print(f"[timer] close '{target_name}': {result}")
        await emit_log(f"STATUS: Closed '{target_name}' after the time limit ({result})")
        self.timer_state = {}

    def stop(self):
        if not self.is_running:
            return False, "Agent is not running."

        self.is_running = False
        if self.is_paused and self._resume_event:
            self._resume_event.set()
        self.is_paused = False  # a stop while paused left the dashboard showing "paused"
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

    @staticmethod
    def _action_expect(action) -> str:
        return action.get("expect", "") if isinstance(action, dict) else ""

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

    @staticmethod
    def _format_last_step(step: int, results: list, overall_changed: bool, still_changing: bool) -> str:
        """Describe the previous step for the next planning call, so the model
        can judge each action against the new screenshot."""
        if not results:
            return ""
        lines = [f"=== PREVIOUS STEP (step {step}) — check these against the CURRENT screenshot ==="]
        for i, (cmd, reason, expect, outcome) in enumerate(results, 1):
            line = f"{i}. {cmd}"
            if reason:
                line += f" — {reason}"
            if expect:
                line += f"\n   expected: {expect}"
            line += f"\n   harness: {outcome}"
            lines.append(line)
        lines.append(f"Whole screen changed during the step: {'yes' if overall_changed else 'no'}")
        if still_changing:
            lines.append("Note: the screen was still animating when this screenshot was taken.")
        lines.append("=== END PREVIOUS STEP ===")
        return "\n".join(lines)

    async def _loop(self, api_key: str, target_type: str, target_name: str, model_name: str, game_instructions: str, emit_log: Callable,
                     provider: str = "gemini_cli", role: str = "gamer", use_grounding: bool = False,
                     grounding_model: str = "", max_budget_usd: float = 0.0,
                     secondary_api_key: str = "", capture_mode: str = "auto"):
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
            llm = LLMIntegration(provider=provider, api_key=api_key if not is_cli_provider(provider) else None,
                                 secondary_api_key=secondary_api_key if not is_cli_provider(provider) else None,
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

        memory_game = self._memory_game(target_type, target_name)
        step = 1
        consecutive_errors = 0
        STEP_HISTORY_MAX = 5  # keep last N step summaries for LLM context
        step_history = []     # list of short summary strings
        last_step_text = ""   # previous step's actions + outcomes, for the model to check
        goals: Optional[GoalTracker] = None
        findings = FindingsLog()
        notebook = RunNotebook()
        prev_step_cmds = []   # commands that led to the current screen (finding repro)
        run_started = time.time()
        last_img_b64 = ""
        last_narration = ""
        pending_zoom = None   # image-space box the model asked to see enlarged (GCA-014)
        click_points = []     # where the previous step's mouse actions landed (GCA-015)
        missed_points = []    # previous step's clicks after which nothing visibly changed (GCA-020)
        streaming = False     # cloud-streaming client detected (GCA-017)
        self.screen_capture.capture_mode = "screen" if capture_mode == "screen" else "auto"
        set_streaming(False)
        await _console_log(f"Findings and report folder: {findings.dir}")

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

        async def _pause(reason: str) -> bool:
            """Pause until the user continues or aborts. Returns False on abort."""
            exec_log.log_pause_reason(reason)
            await _console_log(f"  [!] {reason} — pausing agent")
            await emit_log(f"PAUSED: {reason}. Press Continue Agent or Abort to proceed.")
            self.is_paused = True
            self._resume_event = asyncio.Event()
            await self._resume_event.wait()
            self._resume_event = None
            if not self.is_running:
                exec_log.log_abort()
                await _console_log("Agent aborted by user.")
                return False
            exec_log.log_resume()
            await _console_log("Agent resumed by user.")
            await emit_log("STATUS: Agent resumed — continuing...")
            return True

        while self.is_running:
            try:
                exec_log.log_step_header(step, model_name, target_name)

                # ── Let animations and loading finish before planning (GCA-007) ──
                settled = await self._wait_for_stable(target_type, target_name,
                                                      timeout=SETTLE_TIMEOUT_S, interval=0.4)
                if not settled:
                    last_step_text += ("\nNote: the screen was still changing after "
                                       f"{SETTLE_TIMEOUT_S:.0f}s, so it may be mid-animation or loading.")

                await _console_log(f"Step {step} - Capturing screen ({target_name})...")
                # Capture screen (includes raw PIL image for template matching)
                capture_result = await asyncio.to_thread(self.screen_capture.capture, target_type, target_name)
                img_b64 = capture_result["image"]
                reference_pil = capture_result["pil_image"]
                offset_x = capture_result["offset_x"]
                offset_y = capture_result["offset_y"]
                scale_x = capture_result["scale_x"]
                scale_y = capture_result["scale_y"]
                capture_region = capture_result["region"]
                if capture_result["hwnd"]:
                    set_target_window(capture_result["hwnd"])
                    if step == 1 and capture_mode == "auto" and is_streaming_client(capture_result["hwnd"], target_name):
                        streaming = True
                        set_streaming(True)
                        self.screen_capture.capture_mode = "screen"
                        await _console_log("  [stream] Cloud-streaming client detected — screen capture, slower input timing")
                        await emit_log("STATUS: Streaming client detected — using screen capture and slower input timing")

                # LLM copy: last clicks marked, then rulers (the template-matching
                # reference stays clean). A requested zoom replaces the view.
                view_note = ""
                if pending_zoom:
                    zoom_img, shown, factor = ScreenCapture.zoom_view(
                        capture_result["native_image"], ScreenCapture.native_scale(capture_result["native_image"]),
                        pending_zoom)
                    ruler_img = zoom_img
                    view_note = (f"ZOOMED VIEW: this screenshot shows only the region ({shown[0]},{shown[1]})-"
                                 f"({shown[2]},{shown[3]}) of the screen, enlarged {factor:.1f}x. The ruler labels are "
                                 "FULL-SCREEN coordinates — use them for every command. The next screenshot is the full screen again.")
                    await _console_log(f"  [zoom] showing {shown} at {factor:.1f}x")
                    pending_zoom = None
                else:
                    marked = ScreenCapture.draw_click_markers(reference_pil, click_points)
                    ruler_img = self.screen_capture.draw_rulers(marked)
                    # A click that changed nothing usually missed: show an
                    # enlarged inset around it so the next aim is precise (GCA-020).
                    if missed_points:
                        native = capture_result["native_image"]
                        ruler_img, corner, shown = ScreenCapture.add_miss_inset(
                            ruler_img, native, ScreenCapture.native_scale(native), missed_points[-1])
                        view_note = (f"INSET ({corner} corner, red frame): action {missed_points[-1][0]} of the previous "
                                     f"step clicked but nothing visibly changed. The inset enlarges the area "
                                     f"({shown[0]},{shown[1]})-({shown[2]},{shown[3]}) around that click; its ruler labels are "
                                     "FULL-SCREEN coordinates. Check whether the crosshair is on the intended target and, "
                                     "if not, aim using the inset's rulers.")
                        await _console_log(f"  [inset] zoom inset around missed click {missed_points[-1]} ({corner})")
                img_b64_with_rulers = self.screen_capture.pil_to_base64(ruler_img)

                await _console_log(f"Step {step} - Analyzing with {model_name}... region={capture_region} scale=({scale_x:.2f},{scale_y:.2f})")
                await emit_log(f"THINKING: Step {step} — analyzing screen...")
                # ── Experience / Recipe lookup ──
                step_phash = await asyncio.to_thread(
                    self.screen_capture.capture_phash, target_type, target_name)

                # Try recipe replay first — skip LLM entirely if we have a
                # high-confidence recipe for this screen state.
                matched_recipe = self.recipes.match(step_phash)
                check_text, progress, goal_done, escalated = "", None, False, False
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

                    # ── Goal checklist from the user's instructions (GCA-010) ──
                    if goals is None:
                        goal_texts, ok = await _handle_budget_call(asyncio.to_thread(
                            llm.plan_goals, img_b64_with_rulers, game_instructions, role, model_name))
                        if not ok:
                            break
                        goals = GoalTracker.from_list(goal_texts or [], role)
                        if goals.goals:
                            await _console_log("  [goals]\n" + goals.format_for_prompt())
                            await emit_log("STATUS: Goals — " + " | ".join(g.text for g in goals.goals))
                        else:
                            await _console_log("  [goals] no checklist (planning returned nothing)")

                    # ── Stuck? plan this step with the stronger model (GCA-009) ──
                    escalated = goals.should_escalate()
                    plan_model = escalation_model(provider, model_name) if escalated else model_name
                    stuck_text = goals.stuck_text(step_history) if escalated else ""
                    if escalated:
                        await _console_log(f"  [stuck] {goals.no_progress} steps without progress — planning with {plan_model}")
                        await emit_log(f"STATUS: Stuck — trying a different approach with {plan_model}")

                    # Run the synchronous LLM call in a thread to avoid blocking asyncio loop
                    history_text = "\n".join(step_history) if step_history else ""
                    memories_text = self.memories.format_for_prompt(game=memory_game)
                    second_frame = None
                    if self.dual_frame:
                        await asyncio.sleep(0.4)
                        second_frame = ScreenCapture.pil_to_base64(await asyncio.to_thread(
                            self.screen_capture.capture_fresh, target_type, target_name))
                    t_llm = time.time()
                    response, should_continue = await _handle_budget_call(
                        asyncio.to_thread(llm.get_next_action, img_b64_with_rulers, game_instructions, plan_model, role, extra_context,
                                          enabled_tools=active_tools, step_history=history_text, memories_text=memories_text,
                                          last_step=last_step_text, image_size=reference_pil.size,
                                          goals_text="\n\n".join(filter(None, [goals.format_for_prompt(),
                                                                               findings.titles_for_prompt()])),
                                          stuck_text=stuck_text, view_note=view_note,
                                          extra_image_b64=second_frame,
                                          notebook_text=notebook.format_for_prompt())
                    )
                    if not should_continue:
                        break
                    if response is None:
                        continue  # Retry step
                    await _console_log(f"  [llm] planned in {time.time() - t_llm:.1f}s")

                    narration = response.get("narration", "No narration provided.")
                    raw_actions = response.get("actions", [])
                    check_text = response.get("last_step_check", "") or ""
                    progress = response.get("progress")
                    if not isinstance(progress, bool):
                        progress = None
                    goal_done = response.get("goal_done") is True
                    last_img_b64 = img_b64_with_rulers
                    last_narration = narration

                    # ── Run notebook and turn awareness (GCA-021..024) ──
                    for note in notebook.add(response.get("notes_add")):
                        await _console_log(f"  [note] {note}")
                        await emit_log(f"STATUS: Note — {note}")
                    actor_now = str(response.get("actor_now") or "").strip()
                    if actor_now:
                        await _console_log(f"  [actor] {actor_now}")
                        check_text = f"[actor: {actor_now}] " + check_text

                    # ── Goal checklist corrections from the model (GCA-019) ──
                    update = response.get("goals_update") or []
                    if isinstance(update, list) and update and goals.replace_remaining([str(g) for g in update]):
                        await _console_log("  [goals] revised by the model:\n" + goals.format_for_prompt())
                        await emit_log("STATUS: Goals revised — " + " | ".join(
                            g.text for g in goals.goals if g.status in ("current", "pending")))
                        goal_done = False  # the done flag referred to the old goal

                    # ── Findings (GCA-011) ──
                    for raw in (response.get("findings") or [])[:MAX_FINDINGS_PER_STEP]:
                        f = findings.add(raw, step, prev_step_cmds, reference_pil)
                        if f:
                            await _console_log(f"  [finding #{f['id']}] {f['type']}/{f['severity']} "
                                               f"{'confirmed' if f['confirmed'] else 'unconfirmed'}: {f['title']}")
                            await emit_log(f"FINDING: [{f['type']}/{f['severity']}] {f['title']}")

                # Detect LLM error responses (no actions = something went wrong)
                if not raw_actions and ("error" in narration.lower() or "Error" in narration):
                    # Auth / billing errors won't fix themselves by waiting.
                    if re.search(r"\b(401|402|403)\b|authenticat|payment required|credit balance|usage limit|limit reached|out of (extra )?usage",
                                 narration, re.I):
                        await emit_log(f"ERROR: {narration} — stopping (check the API key / billing)")
                        await _console_log(f"  [!] Fatal provider error: {narration[:200]} — stopping agent")
                        break
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
                expects = [self._action_expect(a) for a in raw_actions]

                # Send narration to user-facing telemetry
                if check_text:
                    await _console_log(f"  [check] {'progress' if progress else 'no progress' if progress is False else '?'}: {check_text[:300]}")
                    await emit_log(f"CHECK: {check_text}")
                await emit_log(f"NARRATION: {narration}")
                for (cmd, reason, *_), expect in zip(parsed_actions, expects):
                    label = f"{cmd}" + (f"  —  {reason}" if reason else "")
                    await emit_log(f"ACTION: {label}")

                # Log to markdown
                logger.log_step(step, narration,
                                [(c, r + (f" (expect: {e})" if e else "")) for (c, r, *_), e in zip(parsed_actions, expects)],
                                check=check_text, progress=progress)

                # Snapshot before any actions to detect overall change
                pre_step_pil = await asyncio.to_thread(
                    self.screen_capture.capture_fresh, target_type, target_name)

                # Re-focus game window before executing (focus drifts during LLM call)
                await asyncio.to_thread(self._focus_window, target_type, target_name)

                failed_actions = 0
                blocked_actions = 0
                total_actions = 0
                action_results = []  # (cmd, reason, expect, outcome) for the next prompt
                click_points = []
                missed_points = []

                # Execute actions ONE BY ONE. Clicks/hovers run once — whether
                # they worked is judged by the model on the next screenshot.
                from action_verifier import MIN_CONFIDENCE as VISION_CONF_THRESHOLD
                action_index = 0

                # Phase detection (OCR) already ran on this step's screenshot when
                # the LLM was called; recipe steps run it once here.
                if _active_recipe:
                    current_phase, _ = await asyncio.to_thread(detect_phase, pre_step_pil)

                def _record(outcome: str):
                    cmd_, reason_, *_ = parsed_actions[action_index]
                    action_results.append((cmd_, reason_, expects[action_index] if action_index < len(expects) else "", outcome))

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
                            _record(f"skipped (precondition '{precond}' not met)")
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

                    # ── Save Memory (GCA-006: game-scoped, no coordinates, corrections replace) ──
                    if cmd.startswith("save_memory"):
                        memory_text = cmd.replace("save_memory", "", 1).strip()
                        if memory_text:
                            outcome, _mem = self.memories.save_from_agent(memory_text, game=memory_game)
                            await _console_log(f"  [memory] {outcome}: {memory_text}")
                            await emit_log(f"STATUS: Memory {outcome} — {memory_text}")
                            if outcome.startswith("rejected"):
                                _record(f"memory {outcome}")
                        action_index += 1
                        continue

                    # ── Zoom: next screenshot shows this region enlarged (GCA-014) ──
                    if cmd.lower().startswith("zoom"):
                        try:
                            box = tuple(int(float(v)) for v in cmd.split()[1:5])
                            assert len(box) == 4
                            if abs(box[2] - box[0]) > MAX_ZOOM_BOX_W:
                                # A near-full-screen "zoom" shows nothing new and costs a step.
                                _record(f"not run — zoom box is {abs(box[2] - box[0])} px wide; zoom only on a "
                                        f"small area (at most {MAX_ZOOM_BOX_W} px wide) around the target")
                            else:
                                pending_zoom = box
                                _record("ok — the next screenshot is a zoomed view of that region")
                        except (ValueError, AssertionError):
                            _record("not run (zoom needs x1 y1 x2 y2)")
                        action_index += 1
                        continue

                    # ── click_text: find a visible label with OCR and click it (GCA-016) ──
                    text_note = ""
                    if cmd.lower().startswith("click_text"):
                        label = cmd[len("click_text"):].strip().strip('"\'')
                        native, nscale = await asyncio.to_thread(self.screen_capture.grab_native, target_type, target_name)
                        hit = await asyncio.to_thread(find_text, native, label)
                        if hit is None:
                            await _console_log(f"  [click_text] '{label}' not found on screen")
                            _record(f"not run — OCR could not find the text '{label}' on screen; click it by coordinates")
                            action_index += 1
                            continue
                        hx, hy = round(hit[0] / nscale), round(hit[1] / nscale)
                        text_note = f"; found '{hit[3]}' at ({hx},{hy})"
                        await _console_log(f"  [click_text] '{label}' → '{hit[3]}' at ({hx},{hy}) score {hit[2]:.2f}")
                        cmd = f"click {hx} {hy}"
                        parsed_actions[action_index] = (cmd, reason, *parsed_actions[action_index][2:])

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
                        _record(f"not run (safety filter: {block_reason})")
                        total_actions += 1
                        failed_actions += 1
                        action_index += 1
                        continue

                    # ── Window moved / resized since the screenshot? (GCA-002) ──
                    try:
                        now_region = await asyncio.to_thread(self.screen_capture.current_region, target_type, target_name)
                    except WindowNotFoundError:
                        raise
                    if tuple(now_region) != tuple(capture_region):
                        msg = f"window moved or resized ({capture_region} → {now_region}); replanning"
                        await _console_log(f"  [!] {msg}")
                        exec_log.log(f"  [replan] {msg}")
                        _record(f"not run ({msg})")
                        break

                    action_type = cmd.split()[0].lower()
                    is_drag = action_type == "drag"
                    is_keyboard = action_type in ("press", "hold_key", "release_key", "type")
                    is_wait = action_type == "wait"
                    is_macro = action_type == "run_macro"

                    # `wait` commands always succeed — no screen change expected
                    if is_wait:
                        try:
                            if float(cmd.split()[1]) > MAX_MODEL_WAIT_S:
                                cmd = f"wait {MAX_MODEL_WAIT_S:g}"
                        except (IndexError, ValueError):
                            cmd = f"wait {MAX_MODEL_WAIT_S:g}"
                        exec_log.log_exec(0, cmd, 0, offset_x, offset_y, scale_x)
                        await asyncio.to_thread(self.input_controller.execute_action, cmd, offset_x, offset_y, scale_x, scale_y)
                        exec_log.log_result(0, True, True)
                        await _console_log(f"  ✓ Wait completed")
                        total_actions += 1
                        exec_log.log_action_outcome(action_index + 1, True, 1)
                        action_index += 1
                        continue

                    # `run_macro` commands bypass vision verification
                    if is_macro:
                        exec_log.log_exec(0, cmd, 0, offset_x, offset_y, scale_x)
                        # Run the macro and then wait for screen stability
                        await asyncio.to_thread(self.input_controller.execute_action, cmd, offset_x, offset_y, scale_x, scale_y)
                        stable = await self._wait_for_stable(target_type, target_name, timeout=4.0, interval=0.4)
                        exec_log.log_result(0, True, stable)
                        await _console_log(f"  ✓ Macro {cmd} completed")
                        _record("ok (macro replayed)")
                        total_actions += 1
                        exec_log.log_action_outcome(action_index + 1, True, 1)
                        action_index += 1
                        continue

                    # ── Keyboard actions: no coordinates to verify ──
                    if is_keyboard:
                        before_pil = await asyncio.to_thread(
                            self.screen_capture.capture_fresh, target_type, target_name)
                        exec_log.log_exec(0, cmd, 0, offset_x, offset_y, scale_x)
                        await _console_log(f"  [exec] {cmd}" + (f"  —  {reason}" if reason else ""))
                        await emit_log({"type": "executing", "command": cmd, "reason": reason, "attempt": 0, "max_retries": 0, "message": f"{cmd}" + (f"  —  {reason}" if reason else "")})
                        status = await asyncio.to_thread(self.input_controller.execute_action, cmd, offset_x, offset_y, scale_x, scale_y)
                        total_actions += 1
                        if status.startswith("blocked"):
                            blocked_actions += 1
                            failed_actions += 1
                            _record(f"NOT SENT — {status}")
                            await emit_log({"type": "warning", "message": f"Input blocked: {status}"})
                            exec_log.log_action_outcome(action_index + 1, False, 1)
                            break
                        await asyncio.sleep(0.3)  # brief pause for game to register
                        after_pil = await asyncio.to_thread(
                            self.screen_capture.capture_fresh, target_type, target_name)
                        changed = ScreenCapture.pil_images_different(
                            before_pil, after_pil, fraction_threshold=CHANGE_HINT_THRESHOLD)
                        _record(f"{status}; screen {'changed' if changed else 'did not visibly change'}")
                        exec_log.log_result(0, changed, True)
                        await _console_log(f"  ✓ Key action sent ({'screen changed' if changed else 'no visible change'})")
                        exec_log.log_action_outcome(action_index + 1, True, 1)
                        action_index += 1
                        continue

                    # ── Mouse actions: one attempt, verifier may correct small drift ──
                    fresh_pil = await asyncio.to_thread(
                        self.screen_capture.capture_fresh, target_type, target_name)
                    if text_note:
                        adjusted_cmd, adjustments = cmd, []  # located on the fresh frame already
                    else:
                        adjusted_cmd, adjustments = await asyncio.to_thread(
                            self.verifier.verify_and_adjust, cmd, reference_pil, fresh_pil, reason)
                    if adjustments:
                        adj_log = self.verifier.format_adjustment_log(adjustments)
                        await _console_log(f"  [vision]\n{adj_log}")
                        exec_log.log_vision(0, adj_log)

                    # ── High-risk classification (OCR; clicks only — hovers/drags/scrolls can't confirm a dialog) ──
                    is_risky, keyword = False, ""
                    if action_type.endswith("click"):
                        is_risky, keyword = await asyncio.to_thread(
                            self.verifier.check_action_risk, adjusted_cmd, fresh_pil)
                    if is_risky and self.unattended:
                        # Timed runs are unattended: don't wait for an approval
                        # nobody will give — refuse the click and tell the model.
                        await _console_log(f"  [!] High-risk keyword '{keyword}' at the target — not clicked (unattended run)")
                        await emit_log(f"WARNING: Refused a click on a '{keyword}' control (unattended run)")
                        _record(f"NOT SENT — the target looks like a destructive control ('{keyword}'); "
                                "choose a safer option (e.g. Cancel / Continue / Back)")
                        total_actions += 1
                        action_index += 1
                        continue
                    if is_risky:
                        await emit_log(f"REQUIRES_APPROVAL: High-risk keyword '{keyword}' detected. Please approve to execute: {adjusted_cmd}")
                        await _console_log(f"  [!] High-risk keyword '{keyword}' detected. Pausing for approval.")
                        self.is_paused = True
                        self._resume_event = asyncio.Event()
                        await self._resume_event.wait()
                        self._resume_event = None
                        if not self.is_running:
                            break
                        await _console_log("  [+] Action approved by user.")
                        await emit_log("STATUS: Action approved — continuing...")

                    await _console_log(f"  [exec] {adjusted_cmd}")
                    await emit_log({"type": "executing", "command": adjusted_cmd, "reason": reason, "attempt": 0, "max_retries": 0, "message": f"{adjusted_cmd}" + (f"  —  {reason}" if reason else "")})

                    before_pil = fresh_pil
                    reference_pil = fresh_pil  # next action's template matching starts from here
                    drag_settle = 0.5
                    exec_log.log_exec(0, adjusted_cmd, drag_settle, offset_x, offset_y, scale_x)
                    status = await asyncio.to_thread(
                        self.input_controller.execute_action,
                        adjusted_cmd, offset_x, offset_y, scale_x, scale_y, drag_settle)
                    total_actions += 1
                    if status.startswith("blocked"):
                        blocked_actions += 1
                        failed_actions += 1
                        _record(f"NOT SENT — {status}")
                        await emit_log({"type": "warning", "message": f"Input blocked: {status}"})
                        exec_log.log_action_outcome(action_index + 1, False, 1)
                        break

                    # Wait for game to process (streams add a round trip)
                    await asyncio.sleep(0.8 if streaming else 0.4)
                    stable = await self._wait_for_stable(target_type, target_name,
                                                          timeout=0.9, interval=0.3)
                    after_pil = await asyncio.to_thread(
                        self.screen_capture.capture_fresh, target_type, target_name)
                    changed = ScreenCapture.pil_images_different(
                        before_pil, after_pil, fraction_threshold=CHANGE_HINT_THRESHOLD)

                    outcome = f"{status}; screen {'changed' if changed else 'did not visibly change'}"
                    # Drag persistence check: one retry with a longer hold if
                    # the dragged item snapped back.
                    if is_drag:
                        await asyncio.sleep(0.6)
                        persist_pil = await asyncio.to_thread(
                            self.screen_capture.capture_fresh, target_type, target_name)
                        if not ScreenCapture.pil_images_different(before_pil, persist_pil, fraction_threshold=CHANGE_HINT_THRESHOLD):
                            exec_log.log("  [result] screen=SNAP_BACK (drag reverted) — retrying once with longer hold")
                            await _console_log("  [!] Drag reverted (snap-back) — retrying once with a longer hold")
                            await asyncio.to_thread(
                                self.input_controller.execute_action,
                                adjusted_cmd, offset_x, offset_y, scale_x, scale_y, 1.2)
                            await asyncio.sleep(1.0)
                            after_pil = await asyncio.to_thread(
                                self.screen_capture.capture_fresh, target_type, target_name)
                            changed = ScreenCapture.pil_images_different(
                                before_pil, after_pil, fraction_threshold=CHANGE_HINT_THRESHOLD)
                            outcome = ("ok after one retry (first drag snapped back)" if changed
                                       else "drag snapped back twice — item probably not placed")
                            if not changed:
                                failed_actions += 1

                    if adjusted_cmd != cmd:
                        outcome += f"; sent as '{adjusted_cmd}' after drift correction"
                    outcome += text_note
                    _record(outcome)
                    try:
                        pts = adjusted_cmd.split()
                        if pts[0] == "drag":
                            click_points.append((action_index + 1, int(pts[3]), int(pts[4])))
                        else:
                            click_points.append((action_index + 1, int(pts[1]), int(pts[2])))
                            if pts[0].endswith("click") and not changed:
                                missed_points.append(click_points[-1])
                    except (IndexError, ValueError):
                        pass
                    exec_log.log_result(0, changed, stable)
                    exec_log.log_action_outcome(action_index + 1, True, 1)
                    await _console_log(f"  → {outcome}")

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
                    # A big UI change, or a next target that no longer matches
                    # the screenshot, means the rest of the plan is stale: stop
                    # and let the next planning call look at the new screen.
                    REPLAN_THRESHOLD = 0.15  # 15% pixel change = major UI shift
                    if action_index < len(parsed_actions) - 1:
                        if ScreenCapture.pil_images_different(pre_step_pil, after_pil, fraction_threshold=REPLAN_THRESHOLD):
                            remaining_count = len(parsed_actions) - action_index - 1
                            await _console_log(f"  [replan] Major screen change detected — discarding {remaining_count} remaining action(s)")
                            await emit_log(f"RE-PLANNING: Screen changed significantly — getting fresh analysis")
                            exec_log.log(f"  [replan] Major change after action {action_index+1}, discarding {remaining_count} remaining action(s)")
                            action_index += 1
                            break

                        next_cmd = parsed_actions[action_index + 1][0]
                        conf = await asyncio.to_thread(
                            self.verifier.check_action_confidence,
                            next_cmd, reference_pil, after_pil)
                        if conf < VISION_CONF_THRESHOLD:
                            remaining_count = len(parsed_actions) - action_index - 1
                            await _console_log(f"  [replan] Next target no longer matches the screen (conf {conf:.2f}) — discarding {remaining_count} remaining action(s)")
                            exec_log.log(f"  [replan] next action confidence {conf:.2f} < {VISION_CONF_THRESHOLD}")
                            action_index += 1
                            break

                    action_index += 1

                # Post-action verification: did anything change at all?
                post_step_pil = await asyncio.to_thread(
                    self.screen_capture.capture_fresh, target_type, target_name)
                overall_changed = ScreenCapture.pil_images_different(pre_step_pil, post_step_pil)

                if not overall_changed:
                    await _console_log(f"  Step {step}: screen identical before/after the actions")
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

                exec_log.log_step_summary(step, total_actions, failed_actions, overall_changed)
                await _console_log(f"  Step {step}: {total_actions - failed_actions}/{total_actions} sent ok, "
                                   f"blocked={blocked_actions}, progress={progress}")

                # ── Goals, stuck handling and pauses (GCA-009/010) ──
                if progress is None:
                    progress_now = overall_changed
                else:
                    progress_now = progress
                if _active_recipe is None and goals is not None:
                    current_goal = goals.current()
                    event = goals.record_step(progress_now, goal_done, escalated)
                    if event == "done" and current_goal:
                        await _console_log(f"  [goals] done: {current_goal.text}")
                        await emit_log(f"STATUS: Goal done — {current_goal.text}")
                    elif event in ("skipped", "budget") and current_goal:
                        await _console_log(f"  [goals] skipped: {current_goal.text} ({current_goal.note})")
                        await emit_log(f"STATUS: Goal skipped — {current_goal.text} ({current_goal.note})")
                        findings.add({"type": "note", "severity": "medium",
                                      "title": f"Could not complete goal: {current_goal.text}",
                                      "details": f"Skipped after {current_goal.steps} steps: {current_goal.note}. "
                                                 f"Last check: {check_text[:300]}",
                                      "confirmed": False}, step, prev_step_cmds, post_step_pil)
                    if goals.all_finished():
                        await _console_log("  [goals] all goals finished — ending the run")
                        await emit_log("STATUS: All goals done or skipped — ending the run")
                        break
                    if event == "pause":
                        if self.unattended:
                            # Nobody is watching a timed run: a pause would just
                            # idle until the limit. Keep going (the stuck streak
                            # was reset, so the stronger model gets another try).
                            await _console_log("  [goals] long no-progress streak — unattended run, continuing")
                            await emit_log("STATUS: Goal still stuck — trying again (unattended run, not pausing)")
                        elif not await _pause("No progress for a long time"):
                            break

                if blocked_actions:
                    if not await _pause("Input blocked: another window is covering the game or has focus"):
                        break

                # ── Step history for LLM context ──
                action_summaries = [f"  {c}" + (f" ({r})" if r else "") + f" → {o}"
                                    for c, r, _e, o in action_results]
                outcome = "screen changed" if overall_changed else "NO VISIBLE CHANGE"
                directive_summary = narration.split('\n')[0][:120] if narration else ""
                check_summary = f" | check: {check_text[:120]}" if check_text else ""
                history_entry = (
                    f"Step {step}: {outcome}{check_summary} | {directive_summary}\n"
                    + "\n".join(action_summaries)
                )
                step_history.append(history_entry)
                if len(step_history) > STEP_HISTORY_MAX:
                    step_history = step_history[-STEP_HISTORY_MAX:]
                last_step_text = self._format_last_step(step, action_results, overall_changed, False)
                prev_step_cmds = [c for c, _r, _e, _o in action_results]

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

            except asyncio.CancelledError:
                await _console_log("Agent task cancelled.")
                break
            except WindowNotFoundError as e:
                await emit_log(f"ERROR: {e} — stopping. Pick the game window again and restart.")
                await _console_log(f"  [!] Target window lost: {e} — stopping agent")
                break
            except Exception as e:
                await emit_log(f"ERROR: {e}")
                await _console_log(f"Exception in agent loop: {e}")
                await asyncio.sleep(5)

        # ── Keep what this run learned for the next run of the same game (GCA-021) ──
        if notebook.notes:
            saved = notebook.save_to_memories(self.memories, memory_game)
            await _console_log(f"Saved {saved} of {len(notebook.notes)} notebook notes as memories for '{memory_game}'")

        # ── End-of-run report (GCA-011) ──
        if last_img_b64 and (findings.findings or (goals and goals.goals)):
            try:
                await _console_log("Writing the end-of-run report...")
                await emit_log("STATUS: Writing the end-of-run report...")
                minutes = (time.time() - run_started) / 60
                session_data = (
                    f"RUN STATS: {step - 1} steps in {minutes:.0f} minutes, role={role}, model={model_name}, "
                    f"cost so far ${llm.cost.total_cost_usd:.3f}\n\n"
                    f"GOALS:\n{goals.summary() if goals else 'none'}\n\n"
                    f"{findings.format_for_report()}\n\n"
                    f"{notebook.format_for_prompt()}\n\n"
                    f"LAST NARRATION: {last_narration[:600]}\n\n"
                    "RECENT STEP HISTORY:\n" + "\n".join(step_history))
                report = await asyncio.to_thread(
                    llm.write_report, last_img_b64, role, game_instructions, session_data, model_name)
                path = findings.save_report(report, role)
                await _console_log(f"Report saved: {path}")
                await emit_log(f"REPORT: {path}")
            except Exception as e:
                await _console_log(f"  [!] Report failed: {e}")

        self.session_state.clear()
        self.is_running = False
        await _console_log("Agent loop terminated.")
