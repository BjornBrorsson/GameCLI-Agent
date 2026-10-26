"""Tests for GCA-009..013: goals, stuck handling, findings and reports. Run from backend/: python -m pytest tests"""
import json
import os
import sys
import types

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import goals as goals_mod  # noqa: E402
from goals import GoalTracker  # noqa: E402
from reporting import FindingsLog  # noqa: E402
import llm as llm_mod  # noqa: E402


def _no_progress(tracker, n, escalated=False):
    events = []
    for _ in range(n):
        events.append(tracker.record_step(progress=False, goal_done=False,
                                          escalated=escalated or tracker.should_escalate()))
    return events


def test_goal_done_advances():
    t = GoalTracker.from_list(["Start a sandbox game", "Build a path"], "reviewer")
    assert t.current().text == "Start a sandbox game"
    assert t.record_step(True, True, False) == "done"
    assert t.current().text == "Build a path"
    assert "[x] 1." in t.format_for_prompt() and "[>] 2." in t.format_for_prompt()


def test_escalates_after_three_no_progress_steps():
    t = GoalTracker.from_list(["Build a piste"], "reviewer")
    _no_progress(t, goals_mod.NO_PROGRESS_ESCALATE)
    assert t.should_escalate()
    assert "YOU ARE STUCK" in t.stuck_text(["Step 1: tried X", "Step 2: tried X", "Step 3: tried X"])


def test_reviewer_skips_after_failed_escalation():
    t = GoalTracker.from_list(["Build a piste", "Build a hut"], "reviewer")
    _no_progress(t, goals_mod.NO_PROGRESS_ESCALATE)
    events = [t.record_step(False, False, escalated=True) for _ in range(goals_mod.ESCALATION_STEPS)]
    assert events[-1] == "skipped"
    assert t.goals[0].status == "skipped" and t.current().text == "Build a hut"


def test_progress_while_escalated_resets():
    t = GoalTracker.from_list(["Build a piste"], "tester")
    _no_progress(t, goals_mod.NO_PROGRESS_ESCALATE)
    assert t.record_step(True, False, escalated=True) == ""
    assert not t.should_escalate()


def test_gamer_never_skips_but_pauses_eventually():
    t = GoalTracker.from_list(["Beat Act 1"], "gamer")
    events = [t.record_step(False, False, escalated=t.should_escalate())
              for _ in range(goals_mod.PAUSE_AFTER_NO_PROGRESS)]
    assert "skipped" not in events and "budget" not in events
    assert events[-1] == "pause"
    assert t.current().text == "Beat Act 1"


def test_step_budget_skips_for_exploration_roles():
    t = GoalTracker.from_list(["Explore menus", "Next"], "explorer", budget=4)
    events = [t.record_step(True, False, False) for _ in range(4)]
    assert events[-1] == "budget"
    assert t.goals[0].status == "skipped"


def test_all_finished():
    t = GoalTracker.from_list(["A"], "reviewer")
    t.record_step(True, True, False)
    assert t.all_finished()


def test_findings_saved_with_screenshot_and_deduped(tmp_path):
    log = FindingsLog(reports_dir=str(tmp_path))
    img = Image.new("RGB", (64, 36))
    f = log.add({"type": "bug", "severity": "high", "title": "Speed label shows %g",
                 "details": "Status line after clicking 3x", "confirmed": True}, 7, ["click 350 20"], img)
    assert f["screenshot"] and os.path.exists(os.path.join(log.dir, f["screenshot"]))
    assert log.add({"type": "bug", "severity": "low", "title": "speed label shows %G"}, 8, [], img) is None
    assert log.add({"type": "weird", "severity": "extreme", "title": "Odd"}, 9, [], None)["type"] == "note"
    saved = json.load(open(os.path.join(log.dir, "findings.json"), encoding="utf-8"))
    assert len(saved) == 2
    text = log.format_for_report()
    assert "CONFIRMED" in text and "unconfirmed" in text and "click 350 20" in text


def test_step_schema_requires_goal_and_findings():
    req = llm_mod.STEP_SCHEMA["required"]
    assert "goal_done" in req and "findings" in req
    assert set(llm_mod.STEP_SCHEMA["properties"]["findings"]["items"]["properties"]["type"]["enum"]) == {
        "bug", "ux", "positive", "note"}


def test_role_prompts_verification_differs():
    assert "reproduce it once" in llm_mod.ROLE_PROMPTS["tester"]
    assert "Do not spend steps reproducing" in llm_mod.ROLE_PROMPTS["reviewer"]


def test_escalation_model_per_provider():
    assert llm_mod.escalation_model("openrouter", "anthropic/claude-haiku-5.5") == "anthropic/claude-sonnet-5.5"
    assert llm_mod.escalation_model("claude_code_cli", "haiku") == "sonnet"
    assert llm_mod.escalation_model("gemini_cli", "gemini-2.5-flash") == "gemini-2.5-flash"


def test_api_provider_uses_one_call_per_step():
    llm = llm_mod.LLMIntegration.__new__(llm_mod.LLMIntegration)
    llm.provider, llm.two_call, llm.turn_count = "openrouter", False, 0
    calls = []

    def fake_api(prompt, image, model, expect_json=True, schema=None):
        calls.append(prompt)
        assert schema is llm_mod.STEP_SCHEMA  # structured output requested
        return {"last_step_check": "", "progress": True, "goal_done": False, "findings": [],
                "narration": "n", "actions": []}
    llm._call_api = fake_api
    llm.get_next_action("aGk=", "Goal", "anthropic/claude-haiku-5.5", role="reviewer",
                        goals_text="=== GOALS === g-marker", stuck_text="stuck-marker", image_size=(1280, 720))
    assert len(calls) == 1
    assert "g-marker" in calls[0] and "stuck-marker" in calls[0]


def test_escalation_capped_per_stuck_streak():
    t = GoalTracker.from_list(["Beat Act 1"], "gamer")
    _no_progress(t, goals_mod.NO_PROGRESS_ESCALATE)
    used = 0
    for _ in range(10):
        esc = t.should_escalate()
        used += esc
        t.record_step(False, False, escalated=esc)
    assert used == goals_mod.ESCALATION_STEPS


def test_model_can_replace_remaining_goals():
    t = GoalTracker.from_list(["Start a match", "Build a Farm", "Destroy the enemy Castle"], "gamer")
    t.record_step(True, True, False)  # "Start a match" done
    assert t.replace_remaining(["Move the hero toward Cat Paradise", "Reach Cat Paradise"])
    assert [g.status for g in t.goals] == ["done", "skipped", "current", "pending"]  # replaced goal kept as history
    assert t.current().text == "Move the hero toward Cat Paradise"
    assert not t.replace_remaining(["Move the hero toward Cat Paradise", "Reach Cat Paradise"])  # no change
    assert "goals_update" in llm_mod.STEP_SCHEMA["required"]
