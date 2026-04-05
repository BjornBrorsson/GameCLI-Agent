"""
SessionState — Save and restore agent session state for crash recovery.

Persists the minimum state needed to resume an interrupted session:
  - step counter
  - configuration (target, model, role, instructions, provider, grounding)
  - experience / recipe counts (stores themselves persist independently)
  - last narration + actions for context

Saved as JSON in backend/.session_state.json (git-ignored).
On startup, the frontend can call GET /api/session to check if a
resumable session exists and offer to continue it.
"""

import json
import os
import time
from typing import Optional, Dict, Any


STATE_FILE = os.path.join(os.path.dirname(__file__), ".session_state.json")


class SessionState:
    """Lightweight session persistence for crash recovery."""

    def __init__(self, filepath: str = STATE_FILE):
        self._filepath = filepath
        self._state: Dict[str, Any] = {}

    def save(self, *,
             step: int,
             target_type: str,
             target_name: str,
             model_name: str,
             game_instructions: str,
             provider: str,
             role: str,
             use_grounding: bool,
             grounding_model: str,
             last_narration: str = "",
             last_actions: list = None):
        """Snapshot current session state to disk."""
        self._state = {
            "step": step,
            "target_type": target_type,
            "target_name": target_name,
            "model_name": model_name,
            "game_instructions": game_instructions,
            "provider": provider,
            "role": role,
            "use_grounding": use_grounding,
            "grounding_model": grounding_model,
            "last_narration": last_narration,
            "last_actions": last_actions or [],
            "saved_at": time.time(),
        }
        try:
            with open(self._filepath, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=1)
        except OSError as e:
            print(f"[session] Failed to save state: {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the last saved session state, if any."""
        if not os.path.isfile(self._filepath):
            return None
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                self._state = json.load(f)
            return dict(self._state)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[session] Failed to load state: {e}")
            return None

    def clear(self):
        """Remove saved state (called on clean stop or abort)."""
        self._state = {}
        try:
            if os.path.isfile(self._filepath):
                os.remove(self._filepath)
        except OSError:
            pass

    @property
    def exists(self) -> bool:
        return os.path.isfile(self._filepath)
