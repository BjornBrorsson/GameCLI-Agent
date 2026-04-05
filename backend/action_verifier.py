"""
ActionVerifier — Local vision feedback loop for coordinate correction.

Before executing each action, this module:
1. Crops a template from the LLM's original screenshot at the action's coordinates
2. Takes a fresh screenshot of the current game state
3. Uses OpenCV template matching to find where the template actually is now
4. Adjusts the action coordinates if the target has moved
5. Progressively expands the search area if the initial match is low-confidence

This handles UI elements shifting after interactions, targets moving,
and general coordinate drift — all without needing another LLM call.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import re
import os

try:
    import pytesseract
    # Auto-discover Tesseract binary
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


# Minimum confidence to accept a template match (TM_CCOEFF_NORMED range: -1 to 1)
MIN_CONFIDENCE = 0.55

# Template half-sizes (radius in pixels, in image space)
TEMPLATE_RADIUS = 48  # 96x96 template — large enough to capture unique element visuals

# Progressive search radii (pixels around expected position)
SEARCH_RADII = [120, 200, 360, None]  # None = full image

# OCR settings
OCR_CROP_RADIUS = 80          # pixels around match point to OCR
OCR_SKIP_THRESHOLD = 0.85     # above this confidence, skip OCR (template is certain enough)


def _pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _ocr_region_text(pil_img: Image.Image, cx: int, cy: int,
                     radius: int = OCR_CROP_RADIUS) -> str:
    """Run Tesseract OCR on a cropped region around (cx, cy).
    Returns extracted text (lowercased), or empty string on failure.
    """
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        w, h = pil_img.size
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w, cx + radius), min(h, cy + radius)
        crop = pil_img.crop((x1, y1, x2, y2))
        text = pytesseract.image_to_string(crop, config="--psm 6")
        return text.strip().lower()
    except Exception:
        return ""


def _extract_intent_keywords(reason: str) -> List[str]:
    """Extract likely game-element names from an action's reason string.
    Finds capitalized words that aren't common English verbs/prepositions.
    Examples: 'Defend', 'Strike', 'Piercing', 'Wail', 'Neutralize'
    """
    if not reason:
        return []
    candidates = re.findall(r'\b[A-Z][a-z]+\+?\b', reason)
    stop = {'Play', 'Click', 'Use', 'The', 'For', 'From', 'With', 'And',
            'This', 'That', 'Its', 'Into', 'Being', 'After', 'Before',
            'All', 'Deal', 'Gain', 'Reduce', 'Kill', 'Finish', 'Ensure',
            'Press', 'End', 'Next', 'Turn', 'Card', 'Hand', 'Button',
            'Extra', 'Target', 'Last', 'Required', 'Incoming', 'Zero',
            'Block', 'Damage', 'Energy', 'Apply', 'Weak', 'Then'}
    return [w for w in candidates if w not in stop and len(w) >= 3]


def _find_all_good_matches(region: np.ndarray, template: np.ndarray,
                           threshold: float = MIN_CONFIDENCE
                           ) -> List[Tuple[int, int, float]]:
    """Find all template match locations above threshold in a region.
    Returns [(cx, cy, confidence), ...] sorted by confidence desc.
    Deduplicates nearby matches.
    """
    th, tw = template.shape[:2]
    rh, rw = region.shape[:2]
    if rh <= th or rw <= tw:
        return []

    result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    locs = np.where(result >= threshold)

    raw = [(int(x + tw // 2), int(y + th // 2), float(result[y, x]))
           for y, x in zip(*locs)]
    raw.sort(key=lambda m: m[2], reverse=True)

    # Deduplicate — keep highest confidence within template-width radius
    filtered: List[Tuple[int, int, float]] = []
    for m in raw:
        if not any(abs(m[0] - f[0]) < tw and abs(m[1] - f[1]) < th
                   for f in filtered):
            filtered.append(m)
    return filtered


def _crop_template(img: np.ndarray, cx: int, cy: int,
                   radius: int = TEMPLATE_RADIUS) -> Optional[np.ndarray]:
    """Crop a square template centered at (cx, cy) from an image.
    Returns None if the template would be mostly out of bounds.
    """
    h, w = img.shape[:2]
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    # Require at least half the template to be in-bounds
    if (x2 - x1) < radius or (y2 - y1) < radius:
        return None
    return img[y1:y2, x1:x2]


def _crop_search_region(img: np.ndarray, cx: int, cy: int,
                        radius: Optional[int]) -> Tuple[np.ndarray, int, int]:
    """Crop a search region centered at (cx, cy) from an image.
    If radius is None, returns the full image.
    Returns (region, offset_x, offset_y) where offsets are the
    top-left corner of the region in the original image.
    """
    if radius is None:
        return img, 0, 0
    h, w = img.shape[:2]
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    return img[y1:y2, x1:x2], x1, y1


def find_template(reference_img: np.ndarray, fresh_img: np.ndarray,
                  cx: int, cy: int) -> Tuple[int, int, float]:
    """Find where a template (cropped from reference_img at cx,cy)
    appears in fresh_img using progressive search expansion.

    Returns (found_x, found_y, confidence).
    If no good match is found, returns original coords with confidence=0.
    """
    template = _crop_template(reference_img, cx, cy)
    if template is None:
        return cx, cy, 0.0

    th, tw = template.shape[:2]
    best_x, best_y, best_conf = cx, cy, 0.0

    for search_radius in SEARCH_RADII:
        region, off_x, off_y = _crop_search_region(
            fresh_img, cx, cy, search_radius)
        rh, rw = region.shape[:2]

        # Region must be larger than template
        if rh <= th or rw <= tw:
            continue

        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_conf:
            # max_loc is top-left of match in the region; convert to center
            # in full image coordinates
            best_x = off_x + max_loc[0] + tw // 2
            best_y = off_y + max_loc[1] + th // 2
            best_conf = max_val

        if best_conf >= MIN_CONFIDENCE:
            break  # good enough, no need to expand

    return best_x, best_y, best_conf


def _parse_coords(action_str: str) -> List[Tuple[int, int]]:
    """Extract (x, y) coordinate pairs from an action string.
    - click x y          → [(x, y)]
    - drag x1 y1 x2 y2   → [(x1, y1), (x2, y2)]
    - other               → []
    """
    parts = action_str.strip().split()
    cmd = parts[0].lower() if parts else ""

    if cmd in ("click", "right_click", "middle_click", "double_click", "hover", "scroll") and len(parts) >= 3:
        return [(int(parts[1]), int(parts[2]))]
    if cmd == "drag" and len(parts) >= 5:
        return [(int(parts[1]), int(parts[2])), (int(parts[3]), int(parts[4]))]
    return []


def _rebuild_action(action_str: str, new_coords: List[Tuple[int, int]]) -> str:
    """Rebuild an action string with adjusted coordinates."""
    parts = action_str.strip().split()
    cmd = parts[0].lower()

    if cmd in ("click", "right_click", "middle_click", "double_click", "hover") and len(new_coords) >= 1:
        rest = parts[3:] if len(parts) > 3 else []
        return f"{cmd} {new_coords[0][0]} {new_coords[0][1]}" + (" ".join([""] + rest) if rest else "")
    if cmd == "scroll" and len(new_coords) >= 1:
        amount = parts[3] if len(parts) > 3 else "1"
        return f"{cmd} {new_coords[0][0]} {new_coords[0][1]} {amount}"
    if cmd == "drag" and len(new_coords) >= 2:
        return f"drag {new_coords[0][0]} {new_coords[0][1]} {new_coords[1][0]} {new_coords[1][1]}"
    return action_str


class ActionVerifier:
    """Verifies and adjusts action coordinates using local template matching."""

    HIGH_RISK_KEYWORDS = ["delete", "abandon", "overwrite", "quit", "reset", "load"]

    def __init__(self):
        self.last_adjustments: List[dict] = []  # log of recent adjustments

    def check_action_risk(self, action_str: str, image: Image.Image) -> Tuple[bool, str]:
        """Check if the action targets a high-risk UI element.
        Uses OCR on the target coordinates to find risky keywords.
        Returns (is_risky, keyword_found).
        """
        if not _TESSERACT_AVAILABLE:
            return False, ""

        coords = _parse_coords(action_str)
        for cx, cy in coords:
            text = _ocr_region_text(image, cx, cy)
            # Ensure text is lowercase for case-insensitive matching,
            # even though _ocr_region_text currently lowers it.
            text_lower = text.lower()
            for kw in self.HIGH_RISK_KEYWORDS:
                if kw.lower() in text_lower:
                    return True, kw
        return False, ""

    def verify_and_adjust(self, action_str: str,
                          reference_pil: Image.Image,
                          fresh_pil: Image.Image,
                          reason: str = "") -> Tuple[str, List[dict]]:
        """Verify action coordinates against a fresh screenshot.

        Args:
            action_str: The original action command (e.g. "drag 290 680 615 430")
            reference_pil: The PIL image the LLM analyzed when deciding the action
            fresh_pil: A fresh PIL screenshot of the current game state
            reason: The LLM's description of the action intent (used for OCR
                    verification of card/element names at matched locations)

        Returns:
            (adjusted_action_str, adjustments_log)
            where adjustments_log is a list of dicts with details per coordinate.
        """
        coords = _parse_coords(action_str)
        if not coords:
            return action_str, []

        ref_cv = _pil_to_cv(reference_pil)
        fresh_cv = _pil_to_cv(fresh_pil)
        keywords = _extract_intent_keywords(reason)

        new_coords = []
        adjustments = []

        for i, (ox, oy) in enumerate(coords):
            label = "start" if i == 0 and len(coords) > 1 else "end" if i == 1 else "target"

            # Use OCR intent verification only for START of drag actions
            # (source element identification) — end points and click targets are
            # typically fixed UI elements where template matching alone is reliable.
            use_ocr = (label == "start" and bool(keywords)
                       and _TESSERACT_AVAILABLE)

            if use_ocr:
                nx, ny, conf, ocr_status = self._find_with_ocr(
                    ref_cv, fresh_cv, fresh_pil, ox, oy, keywords)
            else:
                nx, ny, conf = find_template(ref_cv, fresh_cv, ox, oy)
                ocr_status = None  # OCR not applicable

            dx, dy = nx - ox, ny - oy
            adj = {
                "point": label,
                "original": (ox, oy),
                "adjusted": (nx, ny),
                "delta": (dx, dy),
                "confidence": round(conf, 3),
                "ocr": ocr_status,
            }
            adjustments.append(adj)

            if conf >= MIN_CONFIDENCE:
                new_coords.append((nx, ny))
            else:
                # Low confidence — keep original coordinates
                new_coords.append((ox, oy))

        adjusted_action = _rebuild_action(action_str, new_coords)
        self.last_adjustments = adjustments
        return adjusted_action, adjustments

    def _find_with_ocr(self, ref_cv: np.ndarray, fresh_cv: np.ndarray,
                       fresh_pil: Image.Image, cx: int, cy: int,
                       keywords: List[str]
                       ) -> Tuple[int, int, float, str]:
        """Template matching with OCR intent verification.

        For each candidate match:
        - conf >= 0.85: accept immediately (template is very certain)
        - 0.55 <= conf < 0.85: run OCR — accept if keyword found, else try next
        - No OCR-confirmed match: fall back to best template match

        Returns (x, y, confidence, ocr_status) where ocr_status is one of:
          'verified'  — OCR confirmed the right element
          'skipped'   — template confidence was high enough to skip OCR
          'unverified' — OCR ran but couldn't confirm (using best template match)
          'no_match'  — no template match found at all
        """
        template = _crop_template(ref_cv, cx, cy)
        if template is None:
            return cx, cy, 0.0, "no_match"

        th, tw = template.shape[:2]
        best_x, best_y, best_conf = cx, cy, 0.0

        for search_radius in SEARCH_RADII:
            region, off_x, off_y = _crop_search_region(
                fresh_cv, cx, cy, search_radius)
            all_matches = _find_all_good_matches(region, template)

            for mx, my, conf in all_matches:
                abs_x = off_x + mx
                abs_y = off_y + my

                # Very high confidence — skip OCR
                if conf >= OCR_SKIP_THRESHOLD:
                    return abs_x, abs_y, conf, "skipped"

                # Moderate confidence — verify with OCR
                ocr_text = _ocr_region_text(fresh_pil, abs_x, abs_y)
                if any(kw.lower() in ocr_text for kw in keywords):
                    return abs_x, abs_y, conf, "verified"

                # Track best template match as fallback
                if conf > best_conf:
                    best_x, best_y, best_conf = abs_x, abs_y, conf

            if best_conf >= MIN_CONFIDENCE:
                break

        status = "unverified" if best_conf >= MIN_CONFIDENCE else "no_match"
        return best_x, best_y, best_conf, status

    def check_action_confidence(self, action_str: str,
                               reference_pil: Image.Image,
                               fresh_pil: Image.Image) -> float:
        """Check how well an action's coordinates still match the current screen.
        Returns the minimum template-match confidence across all coordinate
        points (0.0 = no match, 1.0 = perfect match).
        Used to decide whether LLM revalidation is needed before executing.
        """
        coords = _parse_coords(action_str)
        if not coords:
            return 1.0  # no coordinates to check (e.g. key press)

        ref_cv = _pil_to_cv(reference_pil)
        fresh_cv = _pil_to_cv(fresh_pil)

        min_conf = 1.0
        for cx, cy in coords:
            _, _, conf = find_template(ref_cv, fresh_cv, cx, cy)
            min_conf = min(min_conf, conf)
        return min_conf

    @staticmethod
    def nudge_action(action_str: str, attempt: int) -> str:
        """Apply a small coordinate nudge to the action's target point for retries.
        For drags: nudges the END point (the drop target).
        For clicks: nudges the click target.
        Each attempt tries a different direction around the original point.
        """
        # Spiral pattern: up, right, left, down, upper-right, upper-left
        offsets = [
            (0, -18),    # up (hitboxes often higher than visual center)
            (18, 0),     # right
            (-18, 0),    # left
            (0, 18),     # down
            (18, -18),   # upper-right
            (-18, -18),  # upper-left
        ]
        dx, dy = offsets[attempt % len(offsets)]

        coords = _parse_coords(action_str)
        if not coords:
            return action_str

        new_coords = list(coords)
        # Nudge the LAST coordinate (target point)
        last_idx = len(new_coords) - 1
        ox, oy = new_coords[last_idx]
        new_coords[last_idx] = (ox + dx, oy + dy)

        return _rebuild_action(action_str, new_coords)

    def format_adjustment_log(self, adjustments: List[dict]) -> str:
        """Format adjustments into a readable log string."""
        if not adjustments:
            return "  (no coordinates to verify)"
        lines = []
        for adj in adjustments:
            ox, oy = adj["original"]
            nx, ny = adj["adjusted"]
            dx, dy = adj["delta"]
            conf = adj["confidence"]
            status = "✓" if conf >= MIN_CONFIDENCE else "✗ LOW"
            moved = f"Δ({dx:+d},{dy:+d})" if (dx != 0 or dy != 0) else "no shift"
            ocr = adj.get("ocr")
            ocr_tag = f" ocr={ocr}" if ocr is not None else ""
            lines.append(
                f"  [{adj['point']}] ({ox},{oy})→({nx},{ny}) {moved} conf={conf:.2f} {status}{ocr_tag}"
            )
        return "\n".join(lines)
