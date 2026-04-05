"""
Grounding — UI element detection for precise click targeting.

Instead of asking the main LLM to both identify elements AND guess pixel
coordinates in one shot, grounding provides a structured element map FIRST.
The main LLM then references elements by label/ID instead of estimating
coordinates from rulers alone.

Providers:
  - LLMGrounding: Uses a fast LLM call to enumerate visible UI elements.
    No extra dependencies — works with any provider already configured.
  - OmniParserGrounding: Stub for Microsoft OmniParser V2 integration.
    Requires separate installation (PyTorch, model weights).

The grounding step is OPTIONAL. If disabled, the agent falls back to the
existing ruler-based coordinate estimation.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class UIElement:
    """A single detected UI element."""
    id: int
    label: str
    element_type: str  # "button", "text", "icon", "card", "slider", etc.
    x: int  # center x in image coordinates
    y: int  # center y in image coordinates
    width: Optional[int] = None
    height: Optional[int] = None
    confidence: float = 1.0


@dataclass
class GroundingResult:
    """Output of a grounding pass."""
    elements: List[UIElement] = field(default_factory=list)
    raw_text: str = ""  # raw provider output for debugging

    def format_for_prompt(self) -> str:
        """Format element list as text for the LLM action prompt."""
        if not self.elements:
            return ""
        lines = ["DETECTED UI ELEMENTS (use these coordinates for precision):"]
        for el in self.elements:
            size_str = f" ({el.width}x{el.height})" if el.width and el.height else ""
            lines.append(
                f"  [{el.id}] {el.element_type}: \"{el.label}\" at ({el.x}, {el.y}){size_str}"
            )
        lines.append("When clicking an element listed above, use its coordinates directly.")
        return "\n".join(lines)


class GroundingProvider(ABC):
    """Base class for grounding providers."""

    @abstractmethod
    def detect(self, image_b64: str, **kwargs) -> GroundingResult:
        """Detect UI elements in the given screenshot.

        Args:
            image_b64: Base64-encoded JPEG screenshot.

        Returns:
            GroundingResult with detected elements.
        """
        ...


# ── LLM-based grounding ──

GROUNDING_PROMPT = """Analyze this game screenshot and list every visible interactive UI element.
For each element, provide:
- A short label (what the element says or represents)
- The type (button, text_field, icon, card, slider, menu_item, checkbox, tab, health_bar, resource, other)
- The x,y pixel coordinates of its CENTER

Respond with JSON only. No markdown, no backticks.
{
  "elements": [
    {"label": "End Turn", "type": "button", "x": 1150, "y": 680},
    {"label": "Strike", "type": "card", "x": 500, "y": 650}
  ]
}

Be thorough — include ALL clickable/interactive elements visible on screen.
Focus on elements the player might need to interact with.
The image is 1280x720 pixels. Use the yellow rulers on the edges for precision."""


class LLMGrounding(GroundingProvider):
    """Uses the existing LLM to enumerate UI elements in a fast pre-pass.

    This is a lightweight grounding approach that requires no extra
    dependencies. It makes one additional LLM call per step, but uses
    the same provider/API key already configured.
    """

    def __init__(self, llm_integration):
        """
        Args:
            llm_integration: An LLMIntegration instance (from llm.py).
        """
        self._llm = llm_integration

    def detect(self, image_b64: str, model_name: str = "gemini-2.5-flash", **kwargs) -> GroundingResult:
        """Run a fast LLM call to enumerate UI elements."""
        try:
            raw = self._llm._call(GROUNDING_PROMPT, image_b64, model_name)
        except Exception as e:
            print(f"  [grounding] LLM call failed: {e}")
            return GroundingResult()

        if not raw:
            return GroundingResult()

        # Parse response — handle both dict and string
        if isinstance(raw, str):
            raw_text = raw
            # Strip markdown fences if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"  [grounding] Failed to parse LLM response as JSON")
                return GroundingResult(raw_text=raw_text)
        elif isinstance(raw, dict):
            data = raw
            raw_text = json.dumps(raw)
        else:
            return GroundingResult()

        elements = []
        for i, el in enumerate(data.get("elements", [])):
            try:
                elements.append(UIElement(
                    id=i + 1,
                    label=str(el.get("label", "unknown")),
                    element_type=str(el.get("type", "other")),
                    x=int(el["x"]),
                    y=int(el["y"]),
                    width=int(el["width"]) if "width" in el else None,
                    height=int(el["height"]) if "height" in el else None,
                ))
            except (KeyError, ValueError, TypeError):
                continue  # skip malformed entries

        return GroundingResult(elements=elements, raw_text=raw_text)


class OmniParserGrounding(GroundingProvider):
    """Stub for Microsoft OmniParser V2 integration.

    To use this, install OmniParser separately:
      1. Clone https://github.com/microsoft/OmniParser
      2. Download V2 weights
      3. pip install the requirements

    Then implement the detect() method to call the local model.
    """

    def __init__(self, model_path: str = ""):
        self._model_path = model_path
        self._model = None

    def detect(self, image_b64: str, **kwargs) -> GroundingResult:
        """Detect UI elements using OmniParser V2.

        TODO: Implement when OmniParser is installed.
        This requires PyTorch, ultralytics, and the OmniParser weights.
        """
        raise NotImplementedError(
            "OmniParser integration not yet implemented. "
            "Use LLMGrounding or install OmniParser and implement this method."
        )
