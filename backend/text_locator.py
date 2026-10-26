"""
TextLocator — find visible text on screen and return where to click it (GCA-016).

`click_text <label>` lets the model press a button by what it says instead of
estimating pixel coordinates. Pure vision (OCR on the screenshot), so it works
for any game with readable labels.
"""

import difflib
import re
from typing import Optional, Tuple

from PIL import Image, ImageOps

from action_verifier import _TESSERACT_AVAILABLE

try:
    import pytesseract
except ImportError:
    pytesseract = None

MIN_MATCH = 0.72      # fuzzy similarity needed to accept a match
MIN_CONF = 30         # Tesseract word confidence floor


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def _words(img: Image.Image):
    """OCR words with boxes: [(text, left, top, width, height)]."""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 11")
    words = []
    for i, txt in enumerate(data["text"]):
        t = txt.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if t and conf >= MIN_CONF:
            words.append((t, data["left"][i], data["top"][i], data["width"][i], data["height"][i],
                          data["block_num"][i], data["line_num"][i]))
    return words


def find_text(native: Image.Image, label: str) -> Optional[Tuple[int, int, float, str]]:
    """Locate `label` in a full-resolution frame.

    Returns (x, y, score, matched_text) in native pixels — the centre of the
    best matching run of words — or None when nothing matches well enough.
    """
    if not _TESSERACT_AVAILABLE or pytesseract is None:
        return None
    target = _norm(label)
    if not target:
        return None
    n = len(target.split())

    # Downscale very large frames (OCR time grows with pixels; upscaling small
    # frames made Tesseract worse in tests). Try both the normal and inverted
    # image: game UIs often use light text on dark buttons.
    factor = min(1.0, 1600 / max(native.size))
    small = native.convert("L")
    if abs(factor - 1.0) > 0.01:
        small = small.resize((round(native.size[0] * factor), round(native.size[1] * factor)),
                             Image.Resampling.LANCZOS)
    best = None
    for variant in (small, ImageOps.invert(small)):
        words = _words(variant)
        for i in range(len(words)):
            for span in {n, n + 1, max(1, n - 1)}:
                run = words[i:i + span]
                if len(run) < span or len({(w[5], w[6]) for w in run}) > 1:
                    continue  # stay on one OCR line
                text = " ".join(w[0] for w in run)
                score = difflib.SequenceMatcher(None, _norm(text), target).ratio()
                if best is None or score > best[0]:
                    left = min(w[1] for w in run)
                    top = min(w[2] for w in run)
                    right = max(w[1] + w[3] for w in run)
                    bottom = max(w[2] + w[4] for w in run)
                    best = (score, (left + right) / 2, (top + bottom) / 2, text)
    if best is None or best[0] < MIN_MATCH:
        return None
    score, cx, cy, text = best
    return round(cx / factor), round(cy / factor), score, text
