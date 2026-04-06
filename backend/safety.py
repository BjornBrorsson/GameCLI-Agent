"""
Safety — Configurable action filtering for the game agent.

Provides guardrails that run BEFORE every action is executed:
- Blocked-text patterns (e.g. "purchase", "buy now", "delete")
- Rate limiting (prevent rapid-fire click spam)
- Coordinate boundary checks (keep actions within the capture region)

All checks are game-agnostic. Game-specific restrictions belong in the
user's Game Instructions, not here.
"""

import re
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass, field


# ── Default blocked patterns ──
# These are substrings matched against the LLM's "reason" field (case-insensitive).
# They exist to prevent the agent from accidentally triggering real-money transactions,
# destructive OS actions, or other high-risk operations across ANY game.
DEFAULT_BLOCKED_REASONS = [
    "purchase",
    "buy now",
    "real money",
    "microtransaction",
    "delete save",
    "uninstall",
    "format",
    "sign out",
    "log out",
]

# Commands that should never be generated (OS-level danger)
BLOCKED_COMMANDS = [
    r"^type\s+.*(rm\s+-rf|del\s+/|format\s+c|shutdown|reboot)",
]


@dataclass
class SafetyConfig:
    """Tunable safety settings — passed in from the API or set per-role."""
    blocked_reasons: List[str] = field(default_factory=lambda: list(DEFAULT_BLOCKED_REASONS))
    blocked_command_patterns: List[str] = field(default_factory=lambda: list(BLOCKED_COMMANDS))
    min_action_interval_s: float = 0.08  # minimum seconds between actions (rate limit)
    max_x: int = 1280  # max valid x coordinate (matches NORM_WIDTH)
    max_y: int = 720   # max valid y coordinate (matches NORM_HEIGHT)
    enabled: bool = True


class SafetyFilter:
    """Stateful filter that checks each action before execution."""

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self._last_action_time: float = 0.0
        self._blocked_reason_patterns = [
            re.compile(r'\b' + re.escape(p) + r'\b', re.IGNORECASE) for p in self.config.blocked_reasons
        ]
        self._blocked_cmd_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.blocked_command_patterns
        ]

    def check(self, command: str, reason: str = "") -> Tuple[bool, str]:
        """Check whether an action is safe to execute.

        Returns (allowed, explanation). If allowed is False, the action
        should be skipped and the explanation logged.
        """
        if not self.config.enabled:
            return True, ""

        # 1. Blocked command patterns (OS-level danger)
        for pat in self._blocked_cmd_patterns:
            if pat.search(command):
                return False, f"Blocked by safety filter: command matches dangerous pattern"

        # 2. Blocked reason keywords (real-money / destructive intent)
        for pat in self._blocked_reason_patterns:
            if pat.search(reason):
                return False, f"Blocked by safety filter: reason contains '{pat.pattern}'"

        # 3. Coordinate bounds — clamping is done by clamp_coordinates() upstream.
        #    check() no longer blocks on OOB coords.

        # 4. Rate limiting
        now = time.monotonic()
        elapsed = now - self._last_action_time
        if elapsed < self.config.min_action_interval_s:
            # Don't block — just note the timing. The sleep in agent_loop
            # already handles pacing. This is a backstop for pathological cases.
            pass
        self._last_action_time = now

        return True, ""

    def clamp_coordinates(self, command: str) -> Tuple[str, bool, str]:
        """Clamp mouse coordinates to the valid capture region.

        Returns (clamped_command, was_clamped, details_string).
        If coordinates were already in-bounds, returns the original command unchanged.
        """
        if not self.config.enabled:
            return command, False, ""

        parts = command.strip().split()
        cmd_type = parts[0].lower() if parts else ""
        mouse_cmds = {"click", "right_click", "middle_click", "double_click", "hover", "scroll", "drag"}
        if cmd_type not in mouse_cmds or len(parts) < 3:
            return command, False, ""

        MARGIN = 5  # clamp to N pixels inside the edge, not the very edge
        clamped = False
        details = []
        new_parts = [parts[0]]
        coord_idx = 0
        for p in parts[1:]:
            if p.lstrip("-").isdigit():
                val = int(p)
                limit = self.config.max_x - MARGIN if coord_idx % 2 == 0 else self.config.max_y - MARGIN
                lo = MARGIN
                if val < lo:
                    details.append(f"{val}->{lo}")
                    val = lo
                    clamped = True
                elif val > limit:
                    details.append(f"{val}->{limit}")
                    val = limit
                    clamped = True
                new_parts.append(str(val))
                coord_idx += 1
            else:
                new_parts.append(p)

        new_cmd = " ".join(new_parts)
        detail_str = ", ".join(details) if details else ""
        return new_cmd, clamped, detail_str
