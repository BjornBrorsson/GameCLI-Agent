"""
GameState — Game phase detection, OCR-based state extraction, tiered memory,
and win/loss detection for autonomous turn-based game play.

Provides structured game state information to the LLM prompt, dramatically
improving decision quality over raw screenshot analysis alone.
"""

import re
import os
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

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


# ── Game Phases ──

class GamePhase:
    COMBAT = "combat"
    MAP = "map"
    CARD_REWARD = "card_reward"
    REST_SITE = "rest_site"
    SHOP = "shop"
    EVENT = "event"
    MENU = "menu"
    GAME_OVER = "game_over"
    VICTORY = "victory"
    UNKNOWN = "unknown"


# Keywords that indicate specific game phases (case-insensitive OCR matching)
PHASE_KEYWORDS = {
    GamePhase.COMBAT: [
        "end turn", "energy", "block", "intent", "strike", "defend",
        "hp", "draw pile", "discard", "exhaust",
    ],
    GamePhase.MAP: [
        "map", "act ", "floor", "neow", "proceed",
    ],
    GamePhase.CARD_REWARD: [
        "choose a card", "card reward", "pick a card", "add to deck",
        "skip", "bowl",
    ],
    GamePhase.REST_SITE: [
        "rest", "smith", "upgrade", "recall", "dig", "lift",
        "campfire",
    ],
    GamePhase.SHOP: [
        "shop", "purge", "buy", "gold", "price", "sold",
    ],
    GamePhase.EVENT: [
        "choice", "event", "proceed", "leave", "accept", "refuse",
    ],
    GamePhase.GAME_OVER: [
        "defeat", "game over", "you died", "score", "killed by",
        "defect destroyed", "ironclad fell", "silent fell", "watcher fell",
    ],
    GamePhase.VICTORY: [
        "victory", "you win", "congratulations", "score", "heart killed",
        "the end",
    ],
}


# ── Data Classes ──

@dataclass
class CombatState:
    """Extracted state during combat phase."""
    player_hp: Optional[int] = None
    player_max_hp: Optional[int] = None
    player_block: Optional[int] = None
    energy_current: Optional[int] = None
    energy_max: Optional[int] = None
    hand_size: Optional[int] = None
    enemies: List[Dict] = field(default_factory=list)  # [{name, hp, intent}]
    floor_number: Optional[int] = None


@dataclass
class TurnMemory:
    """What happened during the current turn (reset each turn)."""
    cards_played: List[str] = field(default_factory=list)
    energy_spent: int = 0
    damage_dealt: int = 0
    block_gained: int = 0
    turn_number: int = 1


@dataclass
class CombatMemory:
    """What happened during the current combat (reset each combat)."""
    turns_taken: int = 0
    total_damage_dealt: int = 0
    total_damage_taken: int = 0
    cards_seen: List[str] = field(default_factory=list)
    enemies_defeated: List[str] = field(default_factory=list)
    combat_start_hp: Optional[int] = None
    phase_transitions: int = 0  # how many times phase changed during combat


@dataclass
class RunMemory:
    """Persistent memory across the entire run."""
    floor: int = 0
    act: int = 1
    combats_won: int = 0
    combats_lost: int = 0
    deck_notes: str = ""  # LLM-generated summary of deck archetype/strategy
    relics_noted: List[str] = field(default_factory=list)
    run_strategy: str = ""  # LLM-generated strategic direction
    lessons: List[str] = field(default_factory=list)  # post-combat reflections
    max_lessons: int = 10


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


def _ocr_region(pil_img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> str:
    """Run OCR on a specific region. Returns lowercased text."""
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        w, h = pil_img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = pil_img.crop((x1, y1, x2, y2))
        # Upscale small regions for better OCR accuracy
        cw, ch = crop.size
        if cw < 200 or ch < 50:
            scale = max(200 / max(cw, 1), 50 / max(ch, 1), 2.0)
            crop = crop.resize((int(cw * scale), int(ch * scale)), Image.Resampling.LANCZOS)
        text = pytesseract.image_to_string(crop, config="--psm 7")
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
    
    # Combat gets a bonus if we see energy-like patterns (e.g. "3/3" near common positions)
    if best_phase == GamePhase.COMBAT or "end turn" in text_lower:
        best_score = min(1.0, best_score + 0.2)
    
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


# ── State Extraction ──

def extract_combat_state(pil_img: Image.Image) -> CombatState:
    """Extract structured combat state from a screenshot using OCR.
    
    Attempts to read:
    - Player HP (bottom-left area, format: "XX/XX")
    - Energy (bottom-left, format: "X/X" in a circle)
    - Enemy count and basic info
    
    Returns a CombatState with whatever could be extracted.
    """
    state = CombatState()
    w, h = pil_img.size
    
    if not _TESSERACT_AVAILABLE:
        return state
    
    # Player HP — typically bottom-center-left area
    # In Slay the Spire: roughly x=50-200, y=h*0.85-h*0.95
    hp_region = _ocr_region(pil_img, 0, int(h * 0.82), int(w * 0.2), int(h * 0.98))
    hp_match = re.search(r'(\d+)\s*/\s*(\d+)', hp_region)
    if hp_match:
        state.player_hp = int(hp_match.group(1))
        state.player_max_hp = int(hp_match.group(2))
    
    # Energy — typically far bottom-left, a circle with "X/X"
    energy_region = _ocr_region(pil_img, 0, int(h * 0.70), int(w * 0.12), int(h * 0.90))
    energy_match = re.search(r'(\d)\s*/\s*(\d)', energy_region)
    if energy_match:
        state.energy_current = int(energy_match.group(1))
        state.energy_max = int(energy_match.group(2))
    
    # Floor number — typically top-left or top-center
    floor_region = _ocr_region(pil_img, 0, 0, int(w * 0.3), int(h * 0.08))
    floor_match = re.search(r'floor\s*(\d+)', floor_region)
    if floor_match:
        state.floor_number = int(floor_match.group(1))
    
    # Block — shown as a shield icon near the player, bottom-center
    block_region = _ocr_region(pil_img, int(w * 0.15), int(h * 0.75), int(w * 0.35), int(h * 0.90))
    block_match = re.search(r'(\d+)', block_region)
    if block_match and state.player_hp is not None:
        # Only trust block reading if we also found HP (confirms combat screen)
        candidate = int(block_match.group(1))
        if candidate < 200:  # sanity check
            state.player_block = candidate
    
    return state


# ── Tiered Memory Manager ──

class GameMemory:
    """Manages tiered memory: turn → combat → run.
    
    Turn memory resets when a new turn starts.
    Combat memory resets when combat ends (phase changes to non-combat).
    Run memory persists across the entire session.
    """
    
    def __init__(self):
        self.turn = TurnMemory()
        self.combat = CombatMemory()
        self.run = RunMemory()
        self._last_phase = GamePhase.UNKNOWN
        self._combat_active = False
    
    def update_phase(self, new_phase: str, combat_state: Optional[CombatState] = None):
        """Called each step with the detected phase. Manages memory transitions."""
        was_combat = self._combat_active
        is_combat = new_phase == GamePhase.COMBAT
        
        # Combat just started
        if is_combat and not was_combat:
            self._start_combat(combat_state)
        
        # Combat just ended
        if was_combat and not is_combat:
            self._end_combat(new_phase)
        
        self._combat_active = is_combat
        self._last_phase = new_phase
    
    def _start_combat(self, state: Optional[CombatState] = None):
        """Reset combat and turn memory for a new fight."""
        self.combat = CombatMemory()
        self.turn = TurnMemory()
        if state and state.player_hp is not None:
            self.combat.combat_start_hp = state.player_hp
    
    def _end_combat(self, new_phase: str):
        """Archive combat results into run memory."""
        self.run.combats_won += 1
        self.run.floor += 1
    
    def record_turn_end(self):
        """Called when the agent ends its turn."""
        self.combat.turns_taken += 1
        self.turn = TurnMemory(turn_number=self.turn.turn_number + 1)
    
    def record_card_played(self, card_name: str, energy_cost: int = 0):
        """Record a card being played in the current turn."""
        self.turn.cards_played.append(card_name)
        self.turn.energy_spent += energy_cost
        if card_name not in self.combat.cards_seen:
            self.combat.cards_seen.append(card_name)
    
    def record_game_over(self, result: str):
        """Record game over (defeat or victory)."""
        if result == "defeat":
            self.run.combats_lost += 1
        elif result == "victory":
            pass  # Run completed
    
    def add_lesson(self, lesson: str):
        """Add a post-combat reflection to run memory."""
        self.run.lessons.append(lesson)
        if len(self.run.lessons) > self.run.max_lessons:
            self.run.lessons.pop(0)
    
    def update_run_strategy(self, strategy: str):
        """Update the run-level strategic direction."""
        self.run.run_strategy = strategy
    
    def update_deck_notes(self, notes: str):
        """Update notes about deck composition/archetype."""
        self.run.deck_notes = notes
    
    def format_for_prompt(self, phase: str, combat_state: Optional[CombatState] = None) -> str:
        """Format all relevant memory tiers into a text block for the LLM prompt."""
        sections = []
        
        # Run-level context (always included)
        run_parts = []
        if self.run.floor > 0:
            run_parts.append(f"Floor: {self.run.floor}, Act: {self.run.act}")
        if self.run.combats_won > 0:
            run_parts.append(f"Combats won: {self.run.combats_won}")
        if self.run.run_strategy:
            run_parts.append(f"Strategy: {self.run.run_strategy}")
        if self.run.deck_notes:
            run_parts.append(f"Deck: {self.run.deck_notes}")
        if self.run.lessons:
            # Show last 3 lessons
            recent = self.run.lessons[-3:]
            run_parts.append(f"Lessons learned: {'; '.join(recent)}")
        if run_parts:
            sections.append("RUN CONTEXT:\n" + "\n".join(f"  {p}" for p in run_parts))
        
        # Combat-level context (only during combat)
        if phase == GamePhase.COMBAT:
            combat_parts = []
            if self.combat.turns_taken > 0:
                combat_parts.append(f"Combat turn: {self.turn.turn_number} (total turns: {self.combat.turns_taken})")
            if self.combat.combat_start_hp is not None:
                combat_parts.append(f"HP at combat start: {self.combat.combat_start_hp}")
            if self.combat.enemies_defeated:
                combat_parts.append(f"Enemies defeated this combat: {', '.join(self.combat.enemies_defeated)}")
            if combat_parts:
                sections.append("COMBAT CONTEXT:\n" + "\n".join(f"  {p}" for p in combat_parts))
            
            # Turn-level context
            turn_parts = []
            if self.turn.cards_played:
                turn_parts.append(f"Cards played this turn: {', '.join(self.turn.cards_played)}")
                turn_parts.append(f"Energy spent this turn: {self.turn.energy_spent}")
            if turn_parts:
                sections.append("TURN CONTEXT:\n" + "\n".join(f"  {p}" for p in turn_parts))
            
            # OCR-extracted state
            if combat_state:
                state_parts = []
                if combat_state.player_hp is not None:
                    hp_str = f"{combat_state.player_hp}/{combat_state.player_max_hp}" if combat_state.player_max_hp else str(combat_state.player_hp)
                    state_parts.append(f"Player HP: {hp_str}")
                if combat_state.player_block is not None and combat_state.player_block > 0:
                    state_parts.append(f"Block: {combat_state.player_block}")
                if combat_state.energy_current is not None:
                    energy_str = f"{combat_state.energy_current}/{combat_state.energy_max}" if combat_state.energy_max else str(combat_state.energy_current)
                    state_parts.append(f"Energy: {energy_str}")
                if combat_state.floor_number is not None:
                    state_parts.append(f"Floor: {combat_state.floor_number}")
                if state_parts:
                    sections.append("DETECTED STATE (from OCR — verify against screenshot):\n" + "\n".join(f"  {p}" for p in state_parts))
        
        if not sections:
            return ""
        
        return "\n\n".join(sections) + "\n"


# ── Phase-Specific Prompt Addenda ──

PHASE_PROMPTS = {
    GamePhase.COMBAT: """
PHASE: COMBAT
You are in combat. Key priorities:
1. Assess threats: check enemy intents (attack values, debuffs) and your HP/block.
2. Plan energy usage: count your energy and card costs before committing.
3. Play cards in optimal order: debuffs/buffs before attacks, block if needed.
4. You MUST end your turn when done — click End Turn or press the appropriate key.
5. After ending turn, if opponent needs time to act, use 'yield' to wait.
6. Do NOT replay cards you already played this turn (check TURN CONTEXT above).
""",
    GamePhase.MAP: """
PHASE: MAP NAVIGATION
You are on the map screen choosing your next path.
1. Consider your current HP — avoid elites if low HP.
2. Prioritize rest sites if HP is critical (below 30%).
3. Question mark (?) nodes can be events, merchants, or fights.
4. Plan 2-3 nodes ahead, not just the immediate next node.
5. Click your chosen node to proceed.
""",
    GamePhase.CARD_REWARD: """
PHASE: CARD REWARD
You are choosing a card reward after winning combat.
1. Consider your deck archetype and strategy.
2. Avoid taking cards that don't fit your build — a lean deck is often better.
3. Skip if none of the cards improve your deck meaningfully.
4. Click a card to add it, or click Skip to pass.
""",
    GamePhase.REST_SITE: """
PHASE: REST SITE
You are at a campfire rest site.
1. REST (heal 30% max HP) if your HP is below 60%.
2. SMITH (upgrade a card) if your HP is healthy — upgrades are very valuable.
3. Other options (Recall, Dig, Lift) depend on relics — use them if available and beneficial.
""",
    GamePhase.SHOP: """
PHASE: SHOP
You are in the shop.
1. Card removal (purge) is very valuable — remove Strikes and Defends.
2. Only buy cards that fit your deck strategy.
3. Save gold for critical purchases (relics, removals).
4. Click Leave/Exit when done shopping.
""",
    GamePhase.EVENT: """
PHASE: EVENT
You are at a random event with choices.
1. Read all options carefully before choosing.
2. Consider your current HP and resources when evaluating risks.
3. Some events have hidden outcomes — if unsure, choose the safe option.
""",
    GamePhase.GAME_OVER: """
PHASE: GAME OVER
The game has ended in defeat. Look for a button to return to the main menu,
start a new run, or view the score screen. Click the appropriate button.
""",
    GamePhase.VICTORY: """
PHASE: VICTORY
You won the game! Look for a button to view your score, continue, or return
to the main menu.
""",
}


def get_phase_prompt(phase: str) -> str:
    """Get the phase-specific prompt addendum for the current game phase."""
    return PHASE_PROMPTS.get(phase, "")
