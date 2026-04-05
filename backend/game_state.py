"""
GameState — Game-agnostic phase detection, OCR text extraction, tiered memory,
and win/loss detection for autonomous game play.

This module is deliberately game-agnostic. It provides:
- OCR-based text extraction from screenshots
- Lightweight phase detection using generic gaming keywords
- A three-tier memory system (turn → encounter → session) using free-form observations
- Terminal state detection (game over / victory)

All game-specific knowledge should live in the user's Game Instructions,
NOT in this module.
"""

import re
import os
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from PIL import Image

try:
    import pytesseract
    _TESSERACT_CANDIDATES = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    _TESSERACT_AVAILABLE = False
    for _p in _TESSERACT_CANDIDATES:
        if os.path.isfile(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            _TESSERACT_AVAILABLE = True
            break
    if not _TESSERACT_AVAILABLE:
        try:
            pytesseract.get_tesseract_version()
            _TESSERACT_AVAILABLE = True
        except Exception:
            pass
except ImportError:
    _TESSERACT_AVAILABLE = False


# ── Game Phases (generic) ──

class GamePhase:
    COMBAT = "combat"
    NAVIGATION = "navigation"
    MENU = "menu"
    DIALOGUE = "dialogue"
    REWARD = "reward"
    SHOP = "shop"
    GAME_OVER = "game_over"
    VICTORY = "victory"
    UNKNOWN = "unknown"


# Keywords that indicate specific game phases (case-insensitive OCR matching).
# These are deliberately generic — they should fire across many games.
PHASE_KEYWORDS = {
    GamePhase.COMBAT: [
        "hp", "health", "attack", "damage", "enemy", "fight",
        "end turn", "battle", "hit", "miss", "block", "shield",
    ],
    GamePhase.NAVIGATION: [
        "map", "proceed", "continue", "next", "travel", "explore",
        "level", "stage", "world", "path",
    ],
    GamePhase.MENU: [
        "menu", "settings", "options", "new game", "load", "save",
        "quit", "resume", "pause",
    ],
    GamePhase.DIALOGUE: [
        "dialogue", "dialog", "choose", "choice", "accept", "refuse",
        "speak", "talk", "reply",
    ],
    GamePhase.REWARD: [
        "reward", "choose", "pick", "loot", "treasure", "item",
        "upgrade", "unlock",
    ],
    GamePhase.SHOP: [
        "shop", "buy", "sell", "price", "gold", "cost", "store",
    ],
    GamePhase.GAME_OVER: [
        "game over", "defeat", "you died", "you lose", "killed",
        "dead", "failed", "try again",
    ],
    GamePhase.VICTORY: [
        "victory", "you win", "congratulations", "completed",
        "the end", "well done",
    ],
}

# Define which tools are available in which game phase. This dynamic tool subsetting
# massively reduces hallucinated actions and token context length.
PHASE_TOOLS = {
    GamePhase.COMBAT: ["press", "click", "wait", "set_objective"],
    GamePhase.NAVIGATION: ["press", "click", "drag", "hold_key", "hover", "wait", "set_objective"],
    GamePhase.MENU: ["press", "click", "scroll", "wait", "set_objective"],
    GamePhase.DIALOGUE: ["press", "click", "wait", "set_objective"],
    GamePhase.REWARD: ["press", "click", "hover", "wait", "set_objective"],
    GamePhase.SHOP: ["press", "click", "scroll", "hover", "wait", "set_objective"],
    GamePhase.GAME_OVER: ["press", "click", "wait", "set_objective"],
    GamePhase.VICTORY: ["press", "click", "wait", "set_objective"],
    GamePhase.UNKNOWN: ["press", "hold_key", "type", "click", "drag", "scroll", "hover", "wait", "set_objective"]
}


# ── Data Classes (game-agnostic) ──

@dataclass
class TurnMemory:
    """What happened during the current turn (reset each turn)."""
    turn_number: int = 1
    actions_taken: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)


@dataclass
class EncounterMemory:
    """What happened during the current encounter/combat (reset when encounter ends)."""
    turns_taken: int = 0
    observations: List[str] = field(default_factory=list)
    outcome: Optional[str] = None  # "won", "lost", or None while ongoing


@dataclass
class SessionMemory:
    """Persistent memory across the entire play session."""
    encounters_won: int = 0
    encounters_lost: int = 0
    strategy: str = ""  # LLM-generated strategic direction
    lessons: List[str] = field(default_factory=list)  # post-encounter reflections
    observations: List[str] = field(default_factory=list)  # notable things seen
    max_lessons: int = 10
    max_observations: int = 20


# ── OCR Helpers ──

def _ocr_full_screen(pil_img: Image.Image) -> str:
    """Run OCR on the full screen image. Returns lowercased text."""
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(pil_img, config="--psm 6")
        return text.strip().lower()
    except Exception:
        return ""


def _extract_numbers(text: str) -> List[int]:
    """Extract all integers from a string."""
    return [int(x) for x in re.findall(r'\d+', text)]


# ── Phase Detection ──

def detect_phase(pil_img: Image.Image, ocr_text: str = "") -> Tuple[str, float]:
    """Detect the current game phase from screenshot OCR text.
    
    Returns (phase, confidence) where confidence is 0.0-1.0.
    Uses keyword matching on OCR output. Falls back to UNKNOWN if
    OCR is unavailable or no keywords match.
    """
    if not ocr_text:
        ocr_text = _ocr_full_screen(pil_img)
    
    if not ocr_text:
        return GamePhase.UNKNOWN, 0.0
    
    text_lower = ocr_text.lower()
    
    # Check terminal states first (highest priority)
    for terminal_phase in [GamePhase.GAME_OVER, GamePhase.VICTORY]:
        keywords = PHASE_KEYWORDS[terminal_phase]
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches >= 2:
            return terminal_phase, min(1.0, matches * 0.4)
    
    # Score each phase by keyword matches
    scores: Dict[str, float] = {}
    for phase, keywords in PHASE_KEYWORDS.items():
        if phase in (GamePhase.GAME_OVER, GamePhase.VICTORY):
            continue
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            scores[phase] = matches / len(keywords)
    
    if not scores:
        return GamePhase.UNKNOWN, 0.0
    
    best_phase = max(scores, key=scores.get)
    best_score = scores[best_phase]
    
    return best_phase, best_score


def detect_game_over(pil_img: Image.Image, ocr_text: str = "") -> Optional[str]:
    """Check if the game is in a terminal state.
    Returns 'defeat', 'victory', or None.
    """
    phase, conf = detect_phase(pil_img, ocr_text)
    if phase == GamePhase.GAME_OVER and conf >= 0.3:
        return "defeat"
    if phase == GamePhase.VICTORY and conf >= 0.3:
        return "victory"
    return None


def extract_screen_text(pil_img: Image.Image) -> str:
    """Extract all readable text from the screenshot via OCR.
    
    Returns the raw OCR text (lowercased). The LLM interprets
    this in the context of whatever game is being played —
    no game-specific parsing is done here.
    """
    return _ocr_full_screen(pil_img)


# ── Tiered Memory Manager ──

class GameMemory:
    """Manages tiered memory: turn → encounter → session.
    
    Turn memory resets when a new turn starts.
    Encounter memory resets when the encounter ends (phase changes away from combat).
    Session memory persists across the entire play session.
    """
    
    def __init__(self):
        self.turn = TurnMemory()
        self.encounter = EncounterMemory()
        self.session = SessionMemory()
        self._last_phase = GamePhase.UNKNOWN
        self._in_encounter = False
    
    def update_phase(self, new_phase: str):
        """Called each step with the detected phase. Manages memory transitions."""
        was_encounter = self._in_encounter
        is_encounter = new_phase == GamePhase.COMBAT
        
        # Encounter just started
        if is_encounter and not was_encounter:
            self._start_encounter()
        
        # Encounter just ended
        if was_encounter and not is_encounter:
            self._end_encounter(new_phase)
        
        self._in_encounter = is_encounter
        self._last_phase = new_phase
    
    def _start_encounter(self):
        """Reset encounter and turn memory for a new encounter."""
        self.encounter = EncounterMemory()
        self.turn = TurnMemory()
    
    def _end_encounter(self, new_phase: str):
        """Archive encounter results into session memory."""
        terminal = new_phase in (GamePhase.GAME_OVER,)
        if terminal:
            self.encounter.outcome = "lost"
            self.session.encounters_lost += 1
        else:
            self.encounter.outcome = "won"
            self.session.encounters_won += 1
    
    def record_turn_end(self):
        """Called when the agent ends its turn."""
        self.encounter.turns_taken += 1
        self.turn = TurnMemory(turn_number=self.turn.turn_number + 1)
    
    def record_action(self, action_description: str):
        """Record an action taken during the current turn."""
        self.turn.actions_taken.append(action_description)
    
    def record_observation(self, observation: str, scope: str = "turn"):
        """Record an observation at the specified scope (turn, encounter, or session)."""
        if scope == "turn":
            self.turn.observations.append(observation)
        elif scope == "encounter":
            self.encounter.observations.append(observation)
        elif scope == "session":
            self.session.observations.append(observation)
            if len(self.session.observations) > self.session.max_observations:
                self.session.observations.pop(0)
    
    def record_game_over(self, result: str):
        """Record game over (defeat or victory)."""
        if result == "defeat":
            # Only increment loss counter if this encounter wasn't already counted as lost
            # (e.g., to avoid double-counting when _end_encounter already handled it)
            if self.encounter.outcome != "lost":
                self.session.encounters_lost += 1
            self.encounter.outcome = "lost"
        elif result == "victory":
            # Victory is already handled by _end_encounter for the final encounter
            # This call is mainly for session-level tracking if needed
            pass
    
    def add_lesson(self, lesson: str):
        """Add a post-encounter reflection to session memory."""
        self.session.lessons.append(lesson)
        if len(self.session.lessons) > self.session.max_lessons:
            self.session.lessons.pop(0)
    
    def update_strategy(self, strategy: str):
        """Update the session-level strategic direction."""
        self.session.strategy = strategy
    
    def format_for_prompt(self, phase: str, ocr_text: str = "") -> str:
        """Format all relevant memory tiers into a text block for the LLM prompt."""
        sections = []
        
        # Session-level context (always included)
        session_parts = []
        if self.session.encounters_won > 0 or self.session.encounters_lost > 0:
            session_parts.append(f"Encounters won: {self.session.encounters_won}, lost: {self.session.encounters_lost}")
        if self.session.strategy:
            session_parts.append(f"Strategy: {self.session.strategy}")
        if self.session.lessons:
            recent = self.session.lessons[-3:]
            session_parts.append(f"Lessons: {'; '.join(recent)}")
        if self.session.observations:
            recent_obs = self.session.observations[-3:]
            session_parts.append(f"Notes: {'; '.join(recent_obs)}")
        if session_parts:
            sections.append("SESSION CONTEXT:\n" + "\n".join(f"  {p}" for p in session_parts))
        
        # Encounter-level context (only during encounters)
        if phase == GamePhase.COMBAT:
            enc_parts = []
            if self.encounter.turns_taken > 0:
                enc_parts.append(f"Encounter turn: {self.turn.turn_number} (total: {self.encounter.turns_taken})")
            if self.encounter.observations:
                recent_obs = self.encounter.observations[-3:]
                enc_parts.append(f"Notes: {'; '.join(recent_obs)}")
            if enc_parts:
                sections.append("ENCOUNTER CONTEXT:\n" + "\n".join(f"  {p}" for p in enc_parts))
            
            # Turn-level context
            turn_parts = []
            if self.turn.actions_taken:
                turn_parts.append(f"Actions this turn: {', '.join(self.turn.actions_taken)}")
            if self.turn.observations:
                turn_parts.append(f"Observations: {'; '.join(self.turn.observations)}")
            if turn_parts:
                sections.append("TURN CONTEXT:\n" + "\n".join(f"  {p}" for p in turn_parts))
        
        # OCR text (if available, let the LLM interpret it)
        if ocr_text:
            # Truncate to avoid token bloat
            truncated = ocr_text[:500]
            if len(ocr_text) > 500:
                truncated += "..."
            sections.append(f"SCREEN TEXT (from OCR — may contain errors):\n  {truncated}")
        
        if not sections:
            return ""
        
        return "\n\n".join(sections) + "\n"


# ── Phase-Specific Prompt Addenda (generic) ──

PHASE_PROMPTS = {
    GamePhase.COMBAT: """
PHASE: COMBAT / ENCOUNTER
You are in an active encounter or battle.
1. Assess the situation: check your health, resources, and what the opponent is doing.
2. Plan your actions before committing — consider what options are available.
3. End your turn when you have no more useful actions to take.
4. After ending your turn, use 'wait' if the game needs time to process.
""",
    GamePhase.NAVIGATION: """
PHASE: NAVIGATION
You are navigating — choosing where to go next.
1. Consider your current state (health, resources) when picking a path.
2. Follow the game objective from your instructions.
3. Click or interact with the destination to proceed.
""",
    GamePhase.MENU: """
PHASE: MENU
You are on a menu screen.
1. Look for the option that advances the game (e.g. New Game, Continue, Start).
2. If this is a settings/options screen, navigate back to gameplay.
""",
    GamePhase.DIALOGUE: """
PHASE: DIALOGUE / EVENT
You are in a dialogue or event with choices.
1. Read all options carefully before choosing.
2. Consider your current state and resources when evaluating risks.
3. Pick the option that best serves your objective.
""",
    GamePhase.REWARD: """
PHASE: REWARD / SELECTION
You are being offered a reward or making a selection.
1. Evaluate each option based on your current strategy and needs.
2. Pick what strengthens your position most — or skip if nothing helps.
""",
    GamePhase.SHOP: """
PHASE: SHOP
You are in a shop or store.
1. Only spend resources on items that meaningfully help your objective.
2. Conserve resources for critical purchases.
3. Leave when done.
""",
    GamePhase.GAME_OVER: """
PHASE: GAME OVER
The game has ended. Look for a button to return to the main menu,
start a new run, or view results. Click the appropriate button.
""",
    GamePhase.VICTORY: """
PHASE: VICTORY
You won! Look for a button to view results, continue, or return
to the main menu.
""",
}


def get_phase_prompt(phase: str) -> str:
    """Get the phase-specific prompt addendum for the current game phase."""
    return PHASE_PROMPTS.get(phase, "")
