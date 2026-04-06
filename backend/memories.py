"""
Memories — Persistent knowledge store for the game agent.

Stores game-specific tips, hotkeys, strategies, and lessons learned.
Memories are injected into the LLM prompt so the agent can leverage
prior knowledge without re-discovering it every session.

Storage: Single JSON file (backend/memories.json), human-readable.
CRUD exposed via REST API; frontend provides a management UI.
"""

import json
import os
import time
import uuid
from typing import List, Optional, Dict, Any


MEMORIES_FILE = os.path.join(os.path.dirname(__file__), "memories.json")


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

    def search(self, game: str = "", tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Filter memories by game and/or tags."""
        results = self._memories
        if game:
            game_lower = game.lower()
            results = [m for m in results
                       if not m.get("game") or game_lower in m["game"].lower()]
        if tags:
            tag_set = {t.lower() for t in tags}
            results = [m for m in results
                       if tag_set & {t.lower() for t in m.get("tags", [])}]
        return results

    def format_for_prompt(self, game: str = "") -> str:
        """Format relevant memories as text for injection into the LLM prompt.
        
        Reloads from disk first so changes from the UI are picked up immediately.
        If a game name is provided, includes:
          - Memories tagged for that specific game
          - Memories with no game tag (universal)
        """
        self._load()  # always read fresh — UI may have added/removed memories
        relevant = self.search(game=game)
        if not relevant:
            return ""

        lines = ["=== AGENT MEMORIES (knowledge from prior sessions) ==="]
        for m in relevant:
            game_tag = f" [{m['game']}]" if m.get("game") else ""
            tag_str = f" (tags: {', '.join(m['tags'])})" if m.get("tags") else ""
            lines.append(f"• {m['content']}{game_tag}{tag_str}")
        lines.append("=== END MEMORIES ===")
        lines.append("Use these memories to inform your decisions. If you discover new useful information (hotkeys, strategies, mechanics), you can save it with the save_memory command.")
        return "\n".join(lines)
