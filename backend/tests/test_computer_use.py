"""Tests for GCA-014..017 computer-use helpers. Run from backend/: python -m pytest tests"""
import os
import sys

import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screen_capture import ScreenCapture, is_streaming_client  # noqa: E402
import text_locator  # noqa: E402
import llm as llm_mod  # noqa: E402


def test_zoom_view_keeps_full_screen_coordinates():
    native = Image.new("RGB", (3840, 2160))
    view, shown, factor = ScreenCapture.zoom_view(native, 3.0, (600, 300, 700, 360))
    assert view.size == (1280, 720)
    x1, y1, x2, y2 = shown
    assert x1 <= 600 and x2 >= 700 and y1 <= 300 and y2 >= 360   # requested box is inside
    assert abs((x2 - x1) / (y2 - y1) - 16 / 9) < 0.05           # screen aspect kept
    assert abs(factor - 1280 / (x2 - x1)) < 0.05


def test_zoom_view_clamps_to_screen_edges():
    native = Image.new("RGB", (1280, 720))
    _, shown, _ = ScreenCapture.zoom_view(native, 1.0, (1250, 700, 1290, 730))
    assert shown[2] <= 1280 and shown[3] <= 720 and shown[0] >= 0 and shown[1] >= 0


def test_default_rulers_unchanged_and_zoom_labels_fine_grained():
    img = Image.new("RGB", (1280, 720))
    assert ScreenCapture.draw_rulers(img).size == (1280, 720)
    # 8x zoom: labels must step in small units so the model can aim within a few pixels
    zoomed = ScreenCapture.draw_rulers(img, origin=(570, 285), units_per_px=1 / 8)
    assert zoomed.size == (1280, 720)


def test_click_markers_draw_red():
    img = ScreenCapture.draw_click_markers(Image.new("RGB", (200, 120)), [(1, 100, 60)])
    r, g, b = img.getpixel((100 + 9, 60))
    assert r > 200 and g < 100


def _font(size):
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


@pytest.mark.skipif(not text_locator._TESSERACT_AVAILABLE, reason="Tesseract not installed")
def test_find_text_on_buttons():
    img = Image.new("RGB", (1280, 720), (20, 30, 50))
    d = ImageDraw.Draw(img)
    for i, label in enumerate(["NEW GAME", "SANDBOX", "LOAD GAME"]):
        y = 300 + i * 60
        d.rectangle([520, y, 760, y + 40], fill=(40, 70, 120))
        d.text((560, y + 8), label, fill=(240, 240, 240), font=_font(22))
    hit = text_locator.find_text(img, "sandbox")
    assert hit is not None
    x, y, score, text = hit
    assert 540 < x < 700 and 360 < y < 400 and score > 0.9
    assert text_locator.find_text(img, "Quit to desktop") is None


def test_streaming_client_detection_by_title():
    assert is_streaming_client(0, "GeForce NOW")
    assert not is_streaming_client(0, "Ski Resort Tycoon (DEBUG)")


def test_prompts_offer_new_commands():
    p = llm_mod.COMBINED_AGENT_PROMPT
    assert "click_text label" in p and "zoom x1 y1 x2 y2" in p and "crosshairs" in p
