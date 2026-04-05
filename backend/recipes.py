"""
Recipes — Replayable action sequences that skip LLM calls.

When the agent successfully completes a multi-step sequence that matches
a saved recipe's trigger (screen hash similarity), it can replay the
recipe's actions directly instead of burning tokens on an LLM round-trip.

Recipes are saved as JSON in backend/Recipes/ (git-ignored).
They can be created automatically from successful experience patterns
or manually by the user.

A recipe has:
  - name: human-readable label ("End Turn", "Open Inventory")
  - trigger_phash: perceptual hash of the screen where the recipe applies
  - actions: list of {"command": "...", "reason": "..."} dicts
  - success_count: how many times this recipe has been replayed successfully
  - fail_count: how many times replay failed (screen didn't change)
  - enabled: whether auto-replay is active
"""

import json
import os
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict


RECIPES_DIR = os.path.join(os.path.dirname(__file__), "Recipes")
RECIPES_FILE = os.path.join(RECIPES_DIR, "recipes.json")


@dataclass
class Recipe:
    """A saved, replayable action sequence."""
    name: str
    trigger_phash: int
    actions: List[Dict[str, str]]
    success_count: int = 0
    fail_count: int = 0
    enabled: bool = True
    created_at: float = 0.0
    last_used: float = 0.0


class RecipeStore:
    """Manages saved recipes for zero-LLM replay."""

    def __init__(self, filepath: str = RECIPES_FILE):
        self._filepath = filepath
        self._recipes: List[Recipe] = []
        self._load()

    def _load(self):
        if not os.path.isfile(self._filepath):
            self._recipes = []
            return
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._recipes = [Recipe(**r) for r in raw]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"[recipes] Failed to load {self._filepath}: {e}")
            self._recipes = []

    def _save(self):
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._recipes], f, indent=1)

    @staticmethod
    def _hash_distance(h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count("1")

    def match(self, phash: int, max_distance: int = 15) -> Optional[Recipe]:
        """Find an enabled recipe whose trigger matches the current screen.

        Uses a tighter threshold than experience recall (15 vs 25) because
        recipes replay without LLM verification — we need high confidence.
        Only returns recipes with a positive track record (success > fail).
        """
        best: Optional[Recipe] = None
        best_dist = max_distance + 1

        for recipe in self._recipes:
            if not recipe.enabled:
                continue
            # Skip recipes that fail more than they succeed
            if recipe.fail_count > recipe.success_count and recipe.success_count > 0:
                continue
            dist = self._hash_distance(phash, recipe.trigger_phash)
            if dist < best_dist:
                best_dist = dist
                best = recipe

        return best

    def record_result(self, recipe: Recipe, succeeded: bool):
        """Update a recipe's track record after replay."""
        if succeeded:
            recipe.success_count += 1
        else:
            recipe.fail_count += 1
        recipe.last_used = time.time()
        self._save()

    def create(self, name: str, trigger_phash: int,
               actions: List[Dict[str, str]]) -> Recipe:
        """Create a new recipe from a successful action sequence."""
        recipe = Recipe(
            name=name,
            trigger_phash=trigger_phash,
            actions=actions,
            created_at=time.time(),
        )
        self._recipes.append(recipe)
        self._save()
        return recipe

    def create_from_experience(self, name: str, phash: int,
                                actions: List[Dict[str, str]],
                                min_occurrences: int = 3,
                                experience_store=None) -> Optional[Recipe]:
        """Promote a frequently-successful experience pattern to a recipe.

        Only creates the recipe if the same pattern has succeeded at least
        `min_occurrences` times in the experience store.
        """
        if experience_store is None:
            return None

        # Check if this pattern already exists as a recipe
        existing = self.match(phash, max_distance=10)
        if existing:
            return None  # already have a recipe for this screen

        # Count how many times similar screens had the same action sequence
        similar = experience_store.recall(phash, max_hash_distance=15, limit=50)
        action_cmds = [a.get("command", "") for a in actions]
        matches = 0
        for entry in similar:
            entry_cmds = [a.get("command", "") for a in entry.actions]
            if entry_cmds == action_cmds:
                matches += 1

        if matches >= min_occurrences:
            return self.create(name, phash, actions)
        return None

    def list_all(self) -> List[Dict]:
        """Return all recipes as dicts (for API/frontend display)."""
        return [asdict(r) for r in self._recipes]

    def toggle(self, index: int, enabled: bool) -> bool:
        """Enable/disable a recipe by index."""
        if 0 <= index < len(self._recipes):
            self._recipes[index].enabled = enabled
            self._save()
            return True
        return False

    def delete(self, index: int) -> bool:
        """Delete a recipe by index."""
        if 0 <= index < len(self._recipes):
            self._recipes.pop(index)
            self._save()
            return True
        return False

    @property
    def recipe_count(self) -> int:
        return len(self._recipes)
