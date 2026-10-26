"""Tests for the run notebook, turn awareness and the optional second frame (GCA-021..025)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notebook import RunNotebook, MAX_NOTES  # noqa: E402
from memories import MemoryStore  # noqa: E402
import llm as llm_mod  # noqa: E402


def test_notebook_dedupes_and_rejects_coordinates():
    nb = RunNotebook()
    added = nb.add(["Units move with two clicks: first shows the path, second confirms.",
                    "Units move with two clicks: the first shows the path, the second confirms.",
                    "The End Turn button is at (1090, 665)",
                    "ok"])
    assert added == ["Units move with two clicks: first shows the path, second confirms."]
    assert "two clicks" in nb.format_for_prompt()


def test_notebook_caps_size():
    import random
    import string
    rnd = random.Random(1)
    words = lambda: " ".join("".join(rnd.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(5))
    notes = [f"note {i}: {words()}" for i in range(MAX_NOTES + 5)]
    nb = RunNotebook()
    nb.add(notes)
    assert len(nb.notes) == MAX_NOTES
    assert nb.notes[0] == notes[5]  # oldest dropped first


def test_notebook_saved_as_game_memories(tmp_path):
    store = MemoryStore(filepath=str(tmp_path / "m.json"))
    nb = RunNotebook()
    nb.add(["Cannot recruit a hero while another hero is in the town."])
    assert nb.save_to_memories(store, "Heroes of Might & Magic III - HD Edition") == 1
    assert "Cannot recruit" in store.format_for_prompt(game="Heroes of Might & Magic III - HD Edition")
    assert nb.save_to_memories(store, "") == 0


def test_schema_and_prompt_have_notes_and_actor():
    req = llm_mod.STEP_SCHEMA["required"]
    assert "notes_add" in req and "actor_now" in req
    p = llm_mod.COMBINED_AGENT_PROMPT
    assert "RUN NOTEBOOK" in p and "actor_now" in p and "right-click" in p


def _api_llm():
    llm = llm_mod.LLMIntegration.__new__(llm_mod.LLMIntegration)
    llm.provider, llm.two_call, llm.turn_count, llm._extra_image = "openrouter", False, 0, None
    return llm


def test_second_frame_and_notebook_reach_the_call():
    llm = _api_llm()
    seen = {}

    def fake_api(prompt, image, model, expect_json=True, schema=None):
        seen["prompt"], seen["extra"] = prompt, llm._extra_image
        return {"narration": "n", "actions": []}
    llm._call_api = fake_api
    llm.get_next_action("aaa=", "Goal", "m", extra_image_b64="bbb=", notebook_text="=== RUN NOTEBOOK === nb-marker",
                        image_size=(1280, 720))
    assert seen["extra"] == "bbb=" and "TWO FRAMES" in seen["prompt"] and "nb-marker" in seen["prompt"]
    assert llm._extra_image is None  # cleared after the call


def test_no_second_frame_by_default():
    llm = _api_llm()
    seen = {}

    def fake_api(prompt, image, model, expect_json=True, schema=None):
        seen["prompt"], seen["extra"] = prompt, llm._extra_image
        return {"narration": "n", "actions": []}
    llm._call_api = fake_api
    llm.get_next_action("aaa=", "Goal", "m", image_size=(1280, 720))
    assert seen["extra"] is None and "TWO FRAMES" not in seen["prompt"]
