"""
Goals — an ordered checklist the agent works through (GCA-010), plus the
"stuck" policy (GCA-009).

The checklist is derived by the model from the user's instructions at the
start of a run, so it stays game agnostic. Each goal has a step budget for
exploration roles; when the agent stops making progress it first plans with a
stronger model, and roles that may skip then move on to the next goal.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# Roles whose job is coverage (finding things), so a goal they cannot finish
# is reported and skipped. Playing roles keep trying instead.
SKIPPING_ROLES = {"tester", "reviewer", "explorer"}

DEFAULT_GOAL_BUDGET = 15       # steps per goal for skipping roles
NO_PROGRESS_ESCALATE = 3       # no-progress steps before using the stronger model
ESCALATION_STEPS = 2           # steps planned by the stronger model before giving up on a goal
PAUSE_AFTER_NO_PROGRESS = 12   # safety pause for roles that never skip


@dataclass
class Goal:
    text: str
    status: str = "pending"     # pending | current | done | skipped
    steps: int = 0
    note: str = ""


@dataclass
class GoalTracker:
    role: str = "gamer"
    budget: int = DEFAULT_GOAL_BUDGET
    goals: List[Goal] = field(default_factory=list)
    no_progress: int = 0        # consecutive no-progress steps on the current goal
    escalated_steps: int = 0    # steps already planned by the stronger model while stuck

    @classmethod
    def from_list(cls, texts: List[str], role: str, budget: int = DEFAULT_GOAL_BUDGET) -> "GoalTracker":
        tracker = cls(role=role, budget=budget, goals=[Goal(t) for t in texts if t.strip()])
        if tracker.goals:
            tracker.goals[0].status = "current"
        return tracker

    @property
    def can_skip(self) -> bool:
        return self.role in SKIPPING_ROLES

    def current(self) -> Optional[Goal]:
        return next((g for g in self.goals if g.status == "current"), None)

    def all_finished(self) -> bool:
        return bool(self.goals) and self.current() is None

    def _advance(self, status: str, note: str = ""):
        g = self.current()
        if g is None:
            return
        g.status, g.note = status, note
        self.no_progress = 0
        self.escalated_steps = 0
        nxt = next((x for x in self.goals if x.status == "pending"), None)
        if nxt is not None:
            nxt.status = "current"

    # ── per-step bookkeeping ──

    def should_escalate(self) -> bool:
        """Plan the next step with the stronger model? Only for the first
        ESCALATION_STEPS stuck steps of a streak — the GFN run used the stronger
        model on 39 steps in a row, which made it most of the cost."""
        return self.no_progress >= NO_PROGRESS_ESCALATE and self.escalated_steps < ESCALATION_STEPS

    def replace_remaining(self, texts: List[str]) -> bool:
        """The model found the checklist doesn't fit the game (e.g. a Farm goal
        in a game without towns): replace the current and pending goals.
        Done and skipped goals stay as history. Returns True if changed."""
        texts = [t.strip() for t in texts if t and t.strip()][:8]
        if not texts:
            return False
        if [g.text for g in self.goals if g.status in ("current", "pending")] == texts:
            return False
        cur = self.current()
        if cur is not None and cur.text not in texts:
            # Keep the abandoned goal in the history (and the report).
            cur.status, cur.note = "skipped", "replaced when the model revised the goals"
        kept = [g for g in self.goals if g.status in ("done", "skipped")]
        self.goals = kept + [Goal(t) for t in texts]
        self.goals[len(kept)].status = "current"
        self.no_progress = 0
        self.escalated_steps = 0
        return True

    def record_step(self, progress: bool, goal_done: bool, escalated: bool) -> str:
        """Update after a planned step. Returns "" or what changed:
        "done", "skipped", "budget" (skipped for running out of steps) or "pause"."""
        g = self.current()
        if g is not None:
            g.steps += 1
        if goal_done:
            self._advance("done")
            return "done"
        self.no_progress = 0 if progress else self.no_progress + 1
        if escalated and not progress:
            self.escalated_steps += 1
        elif progress:
            self.escalated_steps = 0

        if self.can_skip and g is not None:
            if self.escalated_steps >= ESCALATION_STEPS:
                self._advance("skipped", f"no progress after {self.no_progress} steps, "
                                         f"including {ESCALATION_STEPS} with the stronger model")
                return "skipped"
            if g.steps >= self.budget:
                self._advance("skipped", f"step budget ({self.budget}) used up")
                return "budget"
        elif self.no_progress >= PAUSE_AFTER_NO_PROGRESS:
            self.no_progress = 0
            self.escalated_steps = 0
            return "pause"
        return ""

    # ── prompt text ──

    def format_for_prompt(self) -> str:
        if not self.goals:
            return ""
        marks = {"done": "[x]", "current": "[>]", "pending": "[ ]", "skipped": "[-]"}
        lines = ["=== GOALS (work on the [>] goal; set goal_done when it is visibly complete) ==="]
        for i, g in enumerate(self.goals, 1):
            line = f"{marks[g.status]} {i}. {g.text}"
            if g.status == "current" and self.can_skip:
                line += f"  (step {g.steps + 1} of {self.budget})"
            if g.status == "skipped" and g.note:
                line += f"  — skipped: {g.note}"
            lines.append(line)
        lines.append("=== END GOALS ===")
        return "\n".join(lines)

    def stuck_text(self, step_history: List[str]) -> str:
        """Extra instructions for an escalated (stuck) planning step."""
        g = self.current()
        goal = g.text if g else "the current objective"
        recent = "\n".join(step_history[-NO_PROGRESS_ESCALATE:])
        return ("=== YOU ARE STUCK ===\n"
                f"The last {self.no_progress} steps made no progress on: {goal}\n"
                f"Recent attempts:\n{recent}\n"
                "Do NOT repeat these attempts. Look at the screenshot again and choose a clearly different "
                "approach: a different control, menu, input method or order of actions. If the goal looks "
                "impossible from here, record a finding that says why.\n"
                "=== END STUCK ===")

    def summary(self) -> str:
        if not self.goals:
            return "No goal checklist."
        lines = []
        for i, g in enumerate(self.goals, 1):
            line = f"{i}. [{g.status}] {g.text} — {g.steps} step(s)"
            if g.note:
                line += f" ({g.note})"
            lines.append(line)
        return "\n".join(lines)
