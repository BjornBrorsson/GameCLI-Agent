"""
Memories — Persistent knowledge store for the game agent.

Stores game-specific tips, hotkeys, strategies, and lessons learned.
Memories are injected into the LLM prompt so the agent can leverage
prior knowledge without re-discovering it every session.

Storage: Single JSON file (backend/memories.json), human-readable.
CRUD exposed via REST API; frontend provides a management UI.
"""

import difflib
import json
import os
import re
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple


MEMORIES_FILE = os.path.join(os.path.dirname(__file__), "memories.json")

# At most this many memories go into one prompt (newest first).
PROMPT_LIMIT = 10

# A memory this similar to an existing one for the same game replaces it.
SIMILAR_RATIO = 0.85

# "(343, 91)", "at 640,451", "x=600" — pixel positions go stale as soon as a
# window moves, so agent memories that contain them are refused.
_COORD_RE = re.compile(r"\(\s*\d{2,4}\s*,\s*\d{2,4}\s*\)|\b\d{2,4}\s*,\s*\d{2,4}\b|\b[xy]\s*[=~]\s*\d{2,4}\b")
_REPLACES_RE = re.compile(r"^replaces=([0-9a-f]{8})\s+", re.I)


def game_key(name: str) -> str:
    """Normalise a game / window title: 'Ski Resort Tycoon (DEBUG)' -> 'ski resort tycoon'."""
    name = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", name or "")
    return re.sub(r"\s+", " ", name).strip().lower()


def has_coordinates(text: str) -> bool:
    return bool(_COORD_RE.search(text or ""))


class MemoryStore:
    """Manages persistent agent memories."""

    def __init__(self, filepath: str = MEMORIES_FILE):
        self._filepath = filepath
        self._memories: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load memories from disk."""
        if not os.path.isfile(self._filepath):
            self._memories = []
            return
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                self._memories = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[memories] Failed to load {self._filepath}: {e}")
            self._memories = []

    def _save(self):
        """Persist memories to disk (pretty-printed for readability)."""
        os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(self._memories, f, indent=2, ensure_ascii=False)

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all memories (reloads from disk for cross-instance freshness)."""
        self._load()
        return list(self._memories)

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by ID."""
        for m in self._memories:
            if m["id"] == memory_id:
                return m
        return None

    def add(self, content: str, game: str = "", tags: Optional[List[str]] = None,
            source: str = "user") -> Dict[str, Any]:
        """Add a new memory. Returns the created memory dict."""
        memory = {
            "id": str(uuid.uuid4())[:8],
            "content": content,
            "game": game,
            "tags": tags or [],
            "source": source,  # "user" or "agent"
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._memories.append(memory)
        self._save()
        return memory

    def update(self, memory_id: str, content: str = None, game: str = None,
               tags: List[str] = None) -> Optional[Dict[str, Any]]:
        """Update an existing memory. Returns updated memory or None."""
        for m in self._memories:
            if m["id"] == memory_id:
                if content is not None:
                    m["content"] = content
                if game is not None:
                    m["game"] = game
                if tags is not None:
                    m["tags"] = tags
                self._save()
                return m
        return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m["id"] != memory_id]
        if len(self._memories) < before:
            self._save()
            return True
        return False

    def _matches_game(self, m: Dict[str, Any], key: str) -> bool:
        """Memory belongs to game `key`: tagged with it, or (older untagged
        agent memories) names the game in its text. Untagged user memories
        are universal."""
        mg = game_key(m.get("game", ""))
        if mg:
            return bool(key) and (key in mg or mg in key)
        if m.get("source") == "agent":
            return bool(key) and key in m.get("content", "").lower()
        return True

    def search(self, game: str = "", tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Filter memories by game and/or tags."""
        results = self._memories
        if game:
            key = game_key(game)
            results = [m for m in results if self._matches_game(m, key)]
        if tags:
            tag_set = {t.lower() for t in tags}
            results = [m for m in results
                       if tag_set & {t.lower() for t in m.get("tags", [])}]
        return results

    def save_from_agent(self, text: str, game: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Store a memory the agent produced with `save_memory`.

        - `replaces=<id> text` overwrites that memory (a correction).
        - Text with pixel coordinates is refused.
        - Text very similar to an existing memory of the same game replaces it.
        Returns (outcome, memory) with outcome "added" | "replaced" | "rejected: <why>".
        """
        self._load()
        text = (text or "").strip()
        replaces = None
        m = _REPLACES_RE.match(text)
        if m:
            replaces, text = m.group(1).lower(), text[m.end():].strip()
        if not text:
            return "rejected: empty", None
        if has_coordinates(text):
            return "rejected: contains pixel coordinates (save the fact, not the position)", None

        key = game_key(game)
        target = self.get(replaces) if replaces else None
        if target is None:
            for existing in self.search(game=game):
                if game_key(existing.get("game", "")) != key:
                    continue
                if difflib.SequenceMatcher(None, existing["content"].lower(), text.lower()).ratio() >= SIMILAR_RATIO:
                    target = existing
                    break
        if target is not None:
            target["content"] = text
            target["game"] = game
            target["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save()
            return "replaced", target
        return "added", self.add(content=text, game=game, tags=["agent-discovered"], source="agent")

    def format_for_prompt(self, game: str = "", limit: int = PROMPT_LIMIT) -> str:
        """Format relevant memories as text for injection into the LLM prompt.
        
        Reloads from disk first so changes from the UI are picked up immediately.
        Includes memories for this game plus universal (untagged user) ones,
        newest first, capped at `limit`. Agent memories holding pixel
        coordinates are left out: they describe a window layout that is gone.
        """
        self._load()  # always read fresh — UI may have added/removed memories
        relevant = [m for m in self.search(game=game)
                    if not (m.get("source") == "agent" and has_coordinates(m.get("content", "")))]
        relevant.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        relevant = relevant[:limit]
        if not relevant:
            return ""

        lines = ["=== AGENT MEMORIES (knowledge from prior sessions) ==="]
        for m in relevant:
            game_tag = f" [{m['game']}]" if m.get("game") else ""
            lines.append(f"• [{m['id']}] {m['content']}{game_tag}")
        lines.append("=== END MEMORIES ===")
        lines.append("Use these memories to inform your decisions. Save new facts (hotkeys, mechanics, strategies) with save_memory; "
                     "correct a wrong one with save_memory replaces=<id> <corrected text>.")
        return "\n".join(lines)
