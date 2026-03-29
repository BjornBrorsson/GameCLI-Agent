import os
from datetime import datetime

class SessionLogger:
    def __init__(self, narration_dir: str = "Narration"):
        self.narration_dir = narration_dir
        if not os.path.exists(self.narration_dir):
            os.makedirs(self.narration_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.narration_dir, f"Session_{timestamp}.md")
        self.init_log_file()
        
    def init_log_file(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(f"# Agent Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("This file contains the reasoning and narration for each step the agent takes.\n\n")
            f.write("---\n\n")
            
    def log_step(self, step: int, narration: str, actions: list):
        """
        Appends the step's narration and actions to the markdown file.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"### Step {step} — `{ts}`\n")
            f.write(f"**Narration:**\n> {narration}\n\n")
            f.write("**Actions Taken:**\n")
            if not actions:
                f.write("- None\n")
            else:
                for action in actions:
                    if isinstance(action, tuple) and len(action) == 2:
                        cmd, reason = action
                        if reason:
                            f.write(f"- `{cmd}` — {reason}\n")
                        else:
                            f.write(f"- `{cmd}`\n")
                    else:
                        f.write(f"- `{action}`\n")
            f.write("\n---\n\n")

    def get_filename(self):
        return self.filename


class ExecutionLogger:
    """Logs detailed execution events: vision verification, coordinate
    adjustments, retry attempts, action outcomes, error rates, and timing.
    Writes to Logs/Exec_<timestamp>.log as plain text.
    """

    def __init__(self, log_dir: str = "Logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.log_dir, f"Exec_{timestamp}.log")
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(f"=== Execution Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    def log(self, message: str):
        """Append a timestamped line to the execution log."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")

    def log_step_header(self, step: int, model: str, target: str):
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"STEP {step}  |  model={model}  target={target}\n")
            f.write(f"{'='*60}\n")

    def log_action_start(self, index: int, total: int, cmd: str, reason: str):
        self.log(f"ACTION {index}/{total}: {cmd}" + (f"  — {reason}" if reason else ""))

    def log_vision(self, attempt: int, adj_text: str):
        self.log(f"  [vision] attempt={attempt}\n{adj_text}")

    def log_exec(self, attempt: int, cmd: str, settle: float,
                 offset_x: int = 0, offset_y: int = 0, scale: float = 1.0):
        # Compute actual screen coords for diagnostics
        parts = cmd.strip().split()
        screen_info = ""
        try:
            if parts[0] in ("click", "right_click", "middle_click", "double_click") and len(parts) >= 3:
                sx = int(int(parts[1]) * scale) + offset_x
                sy = int(int(parts[2]) * scale) + offset_y
                screen_info = f"  screen=({sx},{sy})"
            elif parts[0] == "drag" and len(parts) >= 5:
                sx1 = int(int(parts[1]) * scale) + offset_x
                sy1 = int(int(parts[2]) * scale) + offset_y
                sx2 = int(int(parts[3]) * scale) + offset_x
                sy2 = int(int(parts[4]) * scale) + offset_y
                screen_info = f"  screen=({sx1},{sy1})->({sx2},{sy2})"
        except (ValueError, IndexError):
            pass
        self.log(f"  [exec] attempt={attempt}  settle={settle:.2f}s  cmd={cmd}{screen_info}")

    def log_result(self, attempt: int, changed: bool, stable: bool):
        status = "CHANGED" if changed else "UNCHANGED"
        stab = "stable" if stable else "still-animating"
        self.log(f"  [result] attempt={attempt}  screen={status}  {stab}")

    def log_action_outcome(self, index: int, succeeded: bool, attempts: int):
        tag = "OK" if succeeded else "FAILED"
        self.log(f"  [{tag}] action {index} after {attempts} attempt(s)")

    def log_step_summary(self, step: int, total: int, failed: int,
                         overall_changed: bool,
                         avg_attempts: float = 1.0, max_attempts: int = 1):
        self.log(f"STEP {step} SUMMARY: {total - failed}/{total} succeeded, "
                 f"{failed} failed, "
                 f"avg_attempts={avg_attempts:.1f}, max_attempts={max_attempts}, "
                 f"overall_screen_change={'yes' if overall_changed else 'no'}")

    def log_revalidation(self, confidence: float, num_updated: int, actions: list):
        """Log a post-action LLM revalidation event."""
        self.log(f"  [revalidate] confidence={confidence:.2f} → LLM update: {num_updated} action(s)")
        for i, (cmd, reason) in enumerate(actions):
            self.log(f"    [{i+1}] {cmd}" + (f" — {reason}" if reason else ""))

    def log_revalidation_skip(self, confidence: float):
        self.log(f"  [revalidate] confidence={confidence:.2f} — coords OK, no update needed")

    def log_retry_assist(self, attempt: int, failed_cmd: str, result_cmd: str, result_reason: str):
        """Log a mid-retry LLM consultation."""
        self.log(f"  [retry-assist] after {attempt} failed attempts on: {failed_cmd}")
        self.log(f"  [retry-assist] LLM says: {result_cmd}" + (f" — {result_reason}" if result_reason else ""))

    def log_pause(self, error_rate_pct: int):
        self.log(f"*** PAUSED — error rate {error_rate_pct}% ***")

    def log_pause_struggle(self, avg_attempts: float, max_attempts: int, failed: int):
        self.log(f"*** PAUSED — struggling: avg_attempts={avg_attempts:.1f}, "
                 f"max_attempts={max_attempts}, failed={failed} ***")

    def log_resume(self):
        self.log(f"*** RESUMED by user ***")

    def log_abort(self):
        self.log(f"*** ABORTED by user ***")

    def get_filename(self):
        return self.filename
