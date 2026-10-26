"""
RunNotebook — facts the model learns during one run, shown in every prompt (GCA-021).

The step history only covers the last few steps, so the agent kept
re-discovering the same rules (two-click movement 8 times in one Heroes 3
run). The notebook keeps short facts for the whole run; at the end they are
saved as game-scoped memories for the next run of the same game.
"""

import difflib
from typing import List

from memories import has_coordinates

MAX_NOTES = 30          # oldest notes drop off beyond this
MAX_NOTE_CHARS = 240
SIMILAR_RATIO = 0.85


class RunNotebook:
    def __init__(self):
        self.notes: List[str] = []

    def add(self, items) -> List[str]:
        """Add new notes from the model. Returns the ones actually added."""
        added = []
        for raw in items or []:
            note = " ".join(str(raw).split())[:MAX_NOTE_CHARS]
            if len(note) < 8 or has_coordinates(note):
                continue
            low = note.lower()
            if any(difflib.SequenceMatcher(None, low, n.lower()).ratio() >= SIMILAR_RATIO for n in self.notes):
                continue
            self.notes.append(note)
            added.append(note)
        if len(self.notes) > MAX_NOTES:
            self.notes = self.notes[-MAX_NOTES:]
        return added

    def format_for_prompt(self) -> str:
        if not self.notes:
            return ("=== RUN NOTEBOOK (empty) ===\nAdd facts you learn with notes_add.\n=== END NOTEBOOK ===")
        lines = ["=== RUN NOTEBOOK (facts learned earlier in this run — follow them) ==="]
        lines += [f"- {n}" for n in self.notes]
        lines.append("=== END NOTEBOOK ===")
        return "\n".join(lines)

    def save_to_memories(self, store, game: str) -> int:
        """Keep the run's notes for the next run of the same game."""
        if not game:
            return 0
        saved = 0
        for n in self.notes:
            outcome, _ = store.save_from_agent(n, game=game)
            if outcome in ("added", "replaced"):
                saved += 1
        return saved
