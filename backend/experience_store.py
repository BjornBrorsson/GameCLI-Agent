"""
ExperienceStore — Cross-session learning for the game agent.

Persists successful action patterns to disk so the agent can recall
what worked in similar situations across sessions. Game-agnostic:
entries are keyed by perceptual hash of the screen state + OCR text,
not by game-specific concepts.

Storage: JSON file in backend/Experience/ (git-ignored).
Lookup: perceptual hash similarity + optional OCR text overlap.
"""

import json
import os
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict


EXPERIENCE_DIR = os.path.join(os.path.dirname(__file__), "Experience")
EXPERIENCE_FILE = os.path.join(EXPERIENCE_DIR, "patterns.json")
MAX_ENTRIES = 500  # cap to prevent unbounded growth


@dataclass
class ExperienceEntry:
    """A single recorded experience."""
    phash: int                          # perceptual hash of the screen
    ocr_snippet: str                    # first 200 chars of OCR text (for secondary matching)
    actions: List[Dict[str, str]]       # [{"command": "...", "reason": "..."}]
    succeeded: bool                     # did the actions achieve the intended effect?
    role: str = "gamer"                 # which role was active
    timestamp: float = 0.0             # when this was recorded


class ExperienceStore:
    """Manages a persistent store of action patterns."""

    def __init__(self, filepath: str = EXPERIENCE_FILE):
        self._filepath = filepath
        self._entries: List[ExperienceEntry] = []
        self._load()

    def _load(self):
        """Load entries from disk."""
        if not os.path.isfile(self._filepath):
            self._entries = []
            return
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._entries = [ExperienceEntry(**e) for e in raw]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"[experience] Failed to load {self._filepath}: {e}")
            self._entries = []

    def _save(self):
        """Persist entries to disk."""
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        # Trim to MAX_ENTRIES (keep most recent)
        if len(self._entries) > MAX_ENTRIES:
            self._entries = self._entries[-MAX_ENTRIES:]
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=1)

    @staticmethod
    def _hash_distance(h1: int, h2: int) -> int:
        """Hamming distance between two perceptual hashes (256-bit)."""
        xor = h1 ^ h2
        return bin(xor).count("1")

    def record(self, phash: int, ocr_snippet: str, actions: List[Dict[str, str]],
               succeeded: bool, role: str = "gamer"):
        """Record an action sequence and whether it succeeded."""
        entry = ExperienceEntry(
            phash=phash,
            ocr_snippet=ocr_snippet[:200],
            actions=actions,
            succeeded=succeeded,
            role=role,
            timestamp=time.time(),
        )
        self._entries.append(entry)
        self._save()

    def recall(self, phash: int, role: str = "gamer",
               max_hash_distance: int = 25,
               limit: int = 3) -> List[ExperienceEntry]:
        """Find past experiences with similar screen states.

        Args:
            phash: Perceptual hash of the current screen.
            role: Only return entries from matching role.
            max_hash_distance: Max Hamming distance for hash similarity (out of 256).
            limit: Max entries to return.

        Returns:
            List of matching ExperienceEntry objects, most recent first.
            Only returns SUCCESSFUL entries.
        """
        matches = []
        for entry in reversed(self._entries):  # most recent first
            if not entry.succeeded:
                continue
            if entry.role != role:
                continue
            dist = self._hash_distance(phash, entry.phash)
            if dist <= max_hash_distance:
                matches.append(entry)
                if len(matches) >= limit:
                    break
        return matches

    def format_for_prompt(self, matches: List[ExperienceEntry]) -> str:
        """Format recalled experiences as context for the LLM prompt."""
        if not matches:
            return ""
        lines = ["PAST EXPERIENCE (similar screens where these actions worked):"]
        for i, entry in enumerate(matches, 1):
            action_strs = [a.get("command", "?") for a in entry.actions]
            lines.append(f"  [{i}] Actions: {', '.join(action_strs)}")
            if entry.ocr_snippet:
                lines.append(f"      Screen text: \"{entry.ocr_snippet[:80]}...\"")
        lines.append("Use this as guidance — the current screen may differ. Verify before reusing.")
        return "\n".join(lines)

    @property
    def entry_count(self) -> int:
        return len(self._entries)
