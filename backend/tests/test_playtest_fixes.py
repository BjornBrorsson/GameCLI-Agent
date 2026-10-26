"""Unit tests for the GCA-001..008 playtest fixes. Run from backend/:  python -m pytest tests"""
import json
import os
import sys
import tempfile
import types

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memories as mem_mod  # noqa: E402
from memories import MemoryStore, has_coordinates, game_key  # noqa: E402
from screen_capture import ScreenCapture  # noqa: E402
import action_verifier  # noqa: E402
import llm as llm_mod  # noqa: E402
from agent_loop import AgentLoop  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return MemoryStore(filepath=str(tmp_path / "memories.json"))


# ── GCA-006: memory hygiene ──

def test_game_key_strips_build_tags():
    assert game_key("Ski Resort Tycoon (DEBUG)") == "ski resort tycoon"
    assert AgentLoop._memory_game("window", "Ski Resort Tycoon (DEBUG)") == "Ski Resort Tycoon"
    assert AgentLoop._memory_game("monitor", "Monitor 1") == ""


@pytest.mark.parametrize("text,expected", [
    ("Inspect window X is at (343, 91)", True),
    ("click 640,451 opens the dialog", True),
    ("hover x=600 shows the tooltip", True),
    ("Path tool is hotkey 1", False),
    ("Fast (3x) and Ultra (8x) speeds", False),
])
def test_coordinate_detection(text, expected):
    assert has_coordinates(text) is expected


def test_coordinate_memory_is_rejected(store):
    outcome, m = store.save_from_agent("The close X sits at (348, 131)", game="Ski Resort Tycoon")
    assert outcome.startswith("rejected") and m is None
    assert store.list_all() == []


def test_replaces_id_overwrites_contradicting_memory(store):
    _, first = store.save_from_agent("Clear All does not clear the notification list", game="Ski Resort Tycoon")
    outcome, second = store.save_from_agent(
        f"replaces={first['id']} Clear All clears the list; the bell resets to 0", game="Ski Resort Tycoon")
    assert outcome == "replaced"
    assert second["id"] == first["id"]
    assert [m["content"] for m in store.list_all()] == ["Clear All clears the list; the bell resets to 0"]


def test_near_duplicate_replaces(store):
    store.save_from_agent("M key cycles map modes: Normal, Height, Slopes, Lifts", game="Ski Resort Tycoon")
    outcome, _ = store.save_from_agent("M key cycles map modes: Normal, Height, Slopes and Lifts", game="Ski Resort Tycoon")
    assert outcome == "replaced"
    assert len(store.list_all()) == 1


def test_prompt_is_game_scoped_and_capped(store):
    for i in range(15):
        store.add(content=f"Ski fact number {i}", game="Ski Resort Tycoon", source="agent")
    store.add(content="Slay the Spire 2: E ends the turn", game="Slay The Spire 2", source="agent")
    store.add(content="Universal user tip", game="", source="user")
    store.add(content="Ski Resort Tycoon: legacy untagged note", game="", source="agent")
    store.add(content="Ski Resort Tycoon: X at (343, 91)", game="Ski Resort Tycoon", source="agent")

    text = store.format_for_prompt(game="Ski Resort Tycoon (DEBUG)")
    bullets = [l for l in text.splitlines() if l.startswith("• ")]
    assert len(bullets) == mem_mod.PROMPT_LIMIT
    assert "Slay the Spire" not in text
    assert "(343, 91)" not in text
    assert all("[" in b for b in bullets)  # ids shown so the model can correct them


def test_legacy_untagged_agent_memory_matches_by_name(store):
    store.add(content="Ski Resort Tycoon: M cycles map modes", game="", source="agent")
    store.add(content="In Slay the Spire 2: E ends turn", game="", source="agent")
    text = store.format_for_prompt(game="Ski Resort Tycoon")
    assert "M cycles map modes" in text and "Slay" not in text


# ── GCA-002: capture keeps the aspect ratio ──

@pytest.mark.parametrize("size,expected", [
    ((1280, 720), (1280, 720)),
    ((3840, 2160), (1280, 720)),
    ((1316, 823), (1151, 720)),   # window rect incl. title bar is no longer squashed
    ((1024, 768), (960, 720)),
])
def test_normalize_keeps_aspect(size, expected):
    img, sx, sy = ScreenCapture._normalize(Image.new("RGB", size))
    assert img.size == expected
    assert abs(sx - size[0] / expected[0]) < 1e-9 and abs(sy - size[1] / expected[1]) < 1e-9


# ── GCA-008: the verifier never jumps far ──

def test_verifier_rejects_large_shift(monkeypatch):
    monkeypatch.setattr(action_verifier, "find_template", lambda ref, fresh, x, y: (x, y - 45, 0.94))
    v = action_verifier.ActionVerifier()
    img = Image.new("RGB", (1280, 720))
    adjusted, adj = v.verify_and_adjust("click 348 131", img, img, "Close the window")
    assert adjusted == "click 348 131"
    assert adj[0]["rejected"] == "too far"


def test_verifier_applies_small_shift(monkeypatch):
    monkeypatch.setattr(action_verifier, "find_template", lambda ref, fresh, x, y: (x + 6, y - 4, 0.9))
    v = action_verifier.ActionVerifier()
    img = Image.new("RGB", (1280, 720))
    adjusted, _ = v.verify_and_adjust("click 348 131", img, img, "Close the window")
    assert adjusted == "click 354 127"


def test_revalidate_prompt_has_command_placeholder():
    assert "{commands_list_short}" in llm_mod.REVALIDATE_PROMPT


# ── GCA-001: the previous step is described for the model ──

def test_last_step_text_lists_expectations_and_outcomes():
    text = AgentLoop._format_last_step(4, [
        ("hover 688 676", "Check ALERTS tooltip", "Tooltip 'Notification Hub' visible", "ok; screen did not visibly change"),
        ("press space", "Pause", "Pause button highlighted", "NOT SENT — blocked: game window is not in the foreground"),
    ], overall_changed=False, still_changing=False)
    assert "expected: Tooltip 'Notification Hub' visible" in text
    assert "NOT SENT" in text
    assert "step 4" in text


# ── GCA-003: Anthropic provider builds a cached, structured request ──

class _FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        usage = types.SimpleNamespace(input_tokens=500, output_tokens=120,
                                      cache_read_input_tokens=3000, cache_creation_input_tokens=0)
        block = types.SimpleNamespace(type="text", text=json.dumps(self.reply))
        return types.SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn", stop_details=None)


def _anthropic_llm(reply):
    llm = llm_mod.LLMIntegration.__new__(llm_mod.LLMIntegration)
    llm.provider = llm_mod.ANTHROPIC_PROVIDER
    llm.api_key = None
    llm.secondary_api_key = None
    llm._using_secondary = False
    llm.cost = llm_mod.CostTracker()
    llm.turn_count = 0
    llm.two_call = False
    llm._anthropic = types.SimpleNamespace(messages=_FakeMessages(reply))
    return llm


def test_anthropic_step_request_shape_and_cost():
    reply = {"last_step_check": "Tooltip visible", "progress": True, "narration": "Hovering tools",
             "actions": [{"command": "hover 100 676", "reason": "Path tooltip", "expect": "Tooltip shows"}]}
    llm = _anthropic_llm(reply)
    out = llm.get_next_action("aGVsbG8=", "Playtest the game", "claude-haiku-5-5", role="tester",
                              memories_text="=== AGENT MEMORIES === mem-marker", last_step="step-marker",
                              image_size=(1280, 720))
    assert out["progress"] is True and out["actions"][0]["expect"] == "Tooltip shows"

    kw = llm._anthropic.messages.kwargs
    assert kw["model"] == "claude-haiku-5-5"
    system_text = kw["system"][0]["text"]
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Playtest the game" in system_text
    # dynamic parts stay out of the cached prefix
    assert "step-marker" not in system_text and "mem-marker" not in system_text
    user = kw["messages"][0]["content"]
    assert user[0]["type"] == "image" and "Screenshot size: 1280x720" in user[1]["text"]
    assert "step-marker" in user[1]["text"] and "mem-marker" in user[1]["text"]
    assert kw["output_config"]["format"]["schema"] == llm_mod.STEP_SCHEMA
    assert llm.cost.cache_read_tokens == 3000
    expected = (500 * 0.10 + 3000 * 0.10 * 0.1 + 120 * 0.50) / 1_000_000
    assert abs(llm.cost.total_cost_usd - expected) < 1e-12


def test_anthropic_static_prompt_is_stable_between_steps():
    reply = {"last_step_check": "", "progress": False, "narration": "", "actions": []}
    llm = _anthropic_llm(reply)
    llm.get_next_action("aGVsbG8=", "Goal", "claude-haiku-5-5", last_step="A", image_size=(1280, 720))
    first = llm._anthropic.messages.kwargs["system"][0]["text"]
    llm.get_next_action("aGVsbG8=", "Goal", "claude-haiku-5-5", last_step="B", image_size=(1280, 720))
    assert llm._anthropic.messages.kwargs["system"][0]["text"] == first
