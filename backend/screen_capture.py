import ctypes
from ctypes import wintypes

# Per-monitor DPI awareness before any window/screen query, so window rects,
# client rects and mss grabs are all in the same physical-pixel space.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import mss
import mss.tools
import pygetwindow as gw
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import math

# Screenshots sent to the LLM fit inside this box. The game's aspect ratio is
# kept (no squashing), so a 16:9 client area becomes exactly 1280x720.
# Coordinates in LLM responses are in the scaled image space and get scaled
# back to native resolution before execution.
NORM_WIDTH  = 1280
NORM_HEIGHT = 720

PW_CLIENTONLY = 1
PW_RENDERFULLCONTENT = 2  # PrintWindow flag: capture DirectX/OpenGL content


class WindowNotFoundError(RuntimeError):
    """The target window is gone, or its title matches no / several windows."""


def find_window(target_name: str):
    """Resolve a window title to a single pygetwindow window.

    An exact title match wins. Otherwise a substring match is accepted only
    when exactly one window matches. Never falls back to another window or
    to the whole monitor.
    """
    wins = [w for w in gw.getAllWindows() if w.title.strip()]
    exact = [w for w in wins if w.title == target_name]
    if exact:
        return exact[0]
    partial = [w for w in wins if target_name.lower() in w.title.lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise WindowNotFoundError(f"No window titled '{target_name}'")
    titles = ", ".join(f"'{w.title}'" for w in partial[:5])
    raise WindowNotFoundError(
        f"'{target_name}' matches several windows ({titles}); use the full title")


def close_window(target_name: str, grace_s: float = 20.0) -> str:
    """Close a window like the user would (WM_CLOSE); if it is still open
    after `grace_s`, end its process. Returns what happened."""
    import time as _time
    try:
        win = find_window(target_name)
    except WindowNotFoundError:
        return "already closed"
    hwnd = win._hWnd
    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
    deadline = _time.time() + grace_s
    while _time.time() < deadline:
        _time.sleep(1.0)
        if not ctypes.windll.user32.IsWindow(hwnd):
            return "closed"
    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = ctypes.windll.kernel32.OpenProcess(0x0001, False, pid.value)  # PROCESS_TERMINATE
    if not handle:
        return "close requested, but the window is still open and its process could not be ended"
    try:
        ok = ctypes.windll.kernel32.TerminateProcess(handle, 0)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    return "process ended" if ok else "close requested, still open"


def client_rect(hwnd) -> tuple:
    """Client area of a window in screen pixels: (left, top, width, height)."""
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, rect.right - rect.left, rect.bottom - rect.top


def _capture_window_gdi(hwnd):
    """Capture a window's client area with PrintWindow, so other windows
    covering it don't end up in the screenshot. Returns None on failure."""
    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return None
        hdc_window = user32.GetDC(hwnd)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
        hbm = gdi32.CreateCompatibleBitmap(hdc_window, w, h)
        gdi32.SelectObject(hdc_mem, hbm)
        res = user32.PrintWindow(hwnd, hdc_mem, PW_CLIENTONLY | PW_RENDERFULLCONTENT)
        if not res:
            res = user32.PrintWindow(hwnd, hdc_mem, PW_CLIENTONLY)
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG),
                ('biHeight', wintypes.LONG), ('biPlanes', wintypes.WORD),
                ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD), ('biXPelsPerMeter', wintypes.LONG),
                ('biYPelsPerMeter', wintypes.LONG), ('biClrUsed', wintypes.DWORD),
                ('biClrImportant', wintypes.DWORD)]
        bmi = BITMAPINFOHEADER()
        bmi.biSize = 40; bmi.biWidth = w; bmi.biHeight = -h; bmi.biPlanes = 1; bmi.biBitCount = 32
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bmi), 0)
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)
        if not res:
            return None
        return Image.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1).convert('RGB')
    except Exception as e:
        print(f"GDI window capture failed: {e}")
        return None


def _is_blank(img: Image.Image) -> bool:
    """True for an (almost) single-colour frame — a failed capture."""
    lo, hi = img.convert("L").getextrema()
    return hi - lo < 4


def window_process_name(hwnd) -> str:
    """Executable file name of the process that owns a window (lower case)."""
    try:
        pid = wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(520)
            size = wintypes.DWORD(520)
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value.replace("/", chr(92)).split(chr(92))[-1].lower()
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass
    return ""


# Cloud-streaming clients: the game is a video stream, so PrintWindow can
# return black or stale frames and input needs more time to arrive (GCA-017).
STREAMING_PROCESSES = ("geforcenow", "geforce now", "xbox", "shadow", "boosteroid", "parsec", "moonlight")


def is_streaming_client(hwnd, title: str = "") -> bool:
    name = window_process_name(hwnd)
    t = (title or "").lower()
    return any(k in name for k in STREAMING_PROCESSES) or "geforce now" in t


class ScreenCapture:
    def __init__(self):
        # "auto": PrintWindow (ignores covering windows), screen grab as fallback.
        # "screen": always grab the screen area (streaming clients, overlays).
        self.capture_mode = "auto"

    @staticmethod
    def _normalize(img: Image.Image, resample=Image.Resampling.LANCZOS):
        """Scale *img* to fit inside NORM_WIDTH x NORM_HEIGHT, keeping its
        aspect ratio.

        Returns (resized_img, scale_x, scale_y) where scale_x/scale_y are
        the multipliers to convert normalised coordinates back to the
        original image size.
        """
        orig_w, orig_h = img.size
        if orig_w == NORM_WIDTH and orig_h == NORM_HEIGHT:
            return img, 1.0, 1.0
        scale = max(orig_w / NORM_WIDTH, orig_h / NORM_HEIGHT)
        new_size = (max(1, round(orig_w / scale)), max(1, round(orig_h / scale)))
        resized = img.resize(new_size, resample)
        return resized, orig_w / new_size[0], orig_h / new_size[1]

    def get_available_sources(self):
        sources = {"monitors": [], "windows": []}

        # Monitors
        with mss.mss() as sct:
            for i, monitor in enumerate(sct.monitors):
                if i == 0:
                    continue # Monitor 0 is "All monitors"
                sources["monitors"].append(f"Monitor {i}")

        # Windows
        windows = gw.getAllWindows()
        for w in windows:
            if w.visible and w.title.strip() != "":
                sources["windows"].append(w.title)

        return sources

    @staticmethod
    def _monitor(sct, target_name: str) -> dict:
        try:
            return sct.monitors[int(target_name.replace("Monitor ", ""))]
        except Exception as e:
            print(f"Error finding monitor: {e}")
            return sct.monitors[1]

    def _grab(self, target_type: str, target_name: str, activate: bool = False):
        """Grab the target at native resolution.

        Returns (img, region, hwnd): region is (left, top, width, height) in
        screen pixels — the client area for windows — and hwnd is None for
        monitors. Raises WindowNotFoundError when the window is missing.
        """
        if target_type == "window":
            win = find_window(target_name)
            hwnd = win._hWnd
            if win.isMinimized:
                win.restore()
            if activate:
                try:
                    win.activate()
                except Exception:
                    pass
            region = client_rect(hwnd)
            if region[2] <= 0 or region[3] <= 0:
                raise WindowNotFoundError(f"Window '{target_name}' has no client area (minimized?)")
            # PrintWindow first: it ignores windows covering the game.
            img = _capture_window_gdi(hwnd) if self.capture_mode != "screen" else None
            if img is None or _is_blank(img):
                left, top, w, h = region
                with mss.mss() as sct:
                    shot = sct.grab({"left": left, "top": top, "width": w, "height": h})
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            return img, region, hwnd

        with mss.mss() as sct:
            mon = self._monitor(sct, target_name)
            shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return img, (mon["left"], mon["top"], mon["width"], mon["height"]), None

    def capture(self, target_type: str, target_name: str) -> dict:
        """
        Captures the target and returns a dict with:
          - "image": base64 encoded JPEG of the scaled image
          - "pil_image": the scaled PIL image the LLM sees
          - "offset_x"/"offset_y": top-left of the captured area in screen coords
          - "scale_x"/"scale_y": native pixels per image pixel
          - "region": (left, top, width, height) of the captured area
          - "hwnd": target window handle (None for monitors)
          - "native_image": the full-resolution frame (for zoomed views)
        """
        native, region, hwnd = self._grab(target_type, target_name, activate=True)
        img, scale_x, scale_y = self._normalize(native)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=70)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image": img_str,
            "pil_image": img,
            "offset_x": region[0],
            "offset_y": region[1],
            "scale_x": scale_x,
            "scale_y": scale_y,
            "region": region,
            "hwnd": hwnd,
            "native_image": native,
        }

    def grab_native(self, target_type: str, target_name: str):
        """Full-resolution frame plus native-pixels-per-image-pixel scale."""
        native, _, _ = self._grab(target_type, target_name)
        return native, self.native_scale(native)

    @staticmethod
    def native_scale(native: Image.Image) -> float:
        """Native pixels per LLM-image pixel (same rule as _normalize)."""
        if native.size == (NORM_WIDTH, NORM_HEIGHT):
            return 1.0
        return max(native.size[0] / NORM_WIDTH, native.size[1] / NORM_HEIGHT)

    @staticmethod
    def zoom_view(native: Image.Image, scale: float, box: tuple):
        """Enlarged view of image-space box (x1, y1, x2, y2) for the model (GCA-014).

        The box is widened to the screen's aspect ratio and at least 160 px wide.
        Returns (view_image, (x1, y1, x2, y2) actually shown, zoom factor). The
        view's rulers are labelled in full-screen image coordinates, so the
        model keeps using the same coordinate space for its commands.
        """
        img_w, img_h = native.size[0] / scale, native.size[1] / scale
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        aspect = img_w / img_h
        w = max(x2 - x1, (y2 - y1) * aspect, 160.0)
        h = w / aspect
        w, h = min(w, img_w), min(h, img_h)
        x1 = min(max(cx - w / 2, 0), img_w - w); y1 = min(max(cy - h / 2, 0), img_h - h)
        x2, y2 = x1 + w, y1 + h
        crop = native.crop((round(x1 * scale), round(y1 * scale), round(x2 * scale), round(y2 * scale)))
        out_w = NORM_WIDTH
        out_h = max(1, round(out_w * h / w))
        view = crop.resize((out_w, out_h), Image.Resampling.LANCZOS)
        factor = out_w / w
        view = ScreenCapture.draw_rulers(view, origin=(x1, y1), units_per_px=1 / factor)
        return view, (round(x1), round(y1), round(x2), round(y2)), factor

    def capture_fresh(self, target_type: str, target_name: str) -> Image.Image:
        """Fast capture returning only the PIL image at the LLM's scale.
        Used by ActionVerifier for template matching — no base64 overhead.
        """
        img, _, _ = self._grab(target_type, target_name)
        # Only used for diffs and template matching: bilinear is several
        # times faster than Lanczos on 4K frames and good enough (GCA-013).
        img, _, _ = self._normalize(img, Image.Resampling.BILINEAR)
        return img

    def current_region(self, target_type: str, target_name: str) -> tuple:
        """Current capture region (client area for windows) without grabbing pixels."""
        if target_type == "window":
            return client_rect(find_window(target_name)._hWnd)
        with mss.mss() as sct:
            mon = self._monitor(sct, target_name)
        return mon["left"], mon["top"], mon["width"], mon["height"]

    def capture_phash(self, target_type: str, target_name: str) -> int:
        """Fast perceptual hash of the current screen.
        Resizes to 16x16 grayscale, thresholds against the mean,
        and returns a 256-bit integer hash. Robust against idle
        animations and minor visual noise — only meaningful changes
        (cards played, enemies dying, UI transitions) flip enough bits.
        """
        img, _, _ = self._grab(target_type, target_name)
        img = img.resize((16, 16), Image.Resampling.BILINEAR).convert("L")
        pixels = list(img.getdata())
        mean = sum(pixels) / len(pixels)
        bits = 0
        for px in pixels:
            bits = (bits << 1) | (1 if px >= mean else 0)
        return bits

    @staticmethod
    def hashes_similar(h1: int, h2: int, threshold: int = 20) -> bool:
        """Compare two perceptual hashes. Returns True if they differ
        by fewer than `threshold` bits (out of 256). Default threshold
        of 20 tolerates idle animations but catches real changes.
        """
        xor = h1 ^ h2
        diff = bin(xor).count("1")
        return diff < threshold

    @staticmethod
    def draw_rulers(img: Image.Image, origin: tuple = (0.0, 0.0), units_per_px: float = 1.0) -> Image.Image:
        """Draw coordinate rulers on the edges of the image.
        Helps LLMs estimate pixel positions by providing visual anchor points.
        Labels are coordinates in the full screenshot: for a normal frame
        (origin 0,0 and 1 unit per pixel) ticks every 50 px, labels every 100;
        a zoomed view (GCA-014) passes the region's origin and scale so its
        labels still read full-screen coordinates.
        Returns a NEW image (does not mutate the original).
        """
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Use a small built-in font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        tick_color = (255, 255, 0)  # yellow — visible on dark and light backgrounds
        label_color = (255, 255, 0)
        shadow_color = (0, 0, 0)
        # Label step in coordinate units: the smallest "nice" step that keeps
        # labels at least ~90 px apart on the drawn image.
        major = next((s for s in (5, 10, 20, 25, 50, 100, 200, 500) if s / units_per_px >= 90), 1000)
        minor = major / 2
        major_len = 12
        minor_len = 6
        ox, oy = origin

        def ticks(start: float, length_px: int):
            v = math.ceil(start / minor) * minor
            while (v - start) / units_per_px < length_px:
                yield v, int(round((v - start) / units_per_px)), abs(v / major - round(v / major)) < 1e-6
                v += minor

        for v, x, is_major in ticks(ox, w):
            tlen = major_len if is_major else minor_len
            draw.line([(x, 0), (x, tlen)], fill=tick_color, width=1)
            draw.line([(x, h - 1), (x, h - 1 - tlen)], fill=tick_color, width=1)
            if is_major and x > 0:
                txt = str(int(round(v)))
                # Shadow for readability (top and bottom rulers)
                draw.text((x + 2, 1), txt, fill=shadow_color, font=font)
                draw.text((x + 1, 0), txt, fill=label_color, font=font)
                draw.text((x + 2, h - 15), txt, fill=shadow_color, font=font)
                draw.text((x + 1, h - 16), txt, fill=label_color, font=font)

        for v, y, is_major in ticks(oy, h):
            tlen = major_len if is_major else minor_len
            draw.line([(0, y), (tlen, y)], fill=tick_color, width=1)
            draw.line([(w - 1, y), (w - 1 - tlen, y)], fill=tick_color, width=1)
            if is_major and y > 0:
                txt = str(int(round(v)))
                draw.text((2, y + 1), txt, fill=shadow_color, font=font)
                draw.text((1, y), txt, fill=label_color, font=font)
                draw.text((w - 30, y + 1), txt, fill=shadow_color, font=font)
                draw.text((w - 31, y), txt, fill=label_color, font=font)

        return img

    @staticmethod
    def add_miss_inset(img: Image.Image, native: Image.Image, scale: float, point: tuple,
                       region_w: int = 128, inset_w: int = 384):
        """Picture-in-picture zoom around a click that changed nothing (GCA-020).

        `img` is the model's (already marked and ruled) screenshot; `point` is
        (number, x, y) in image coordinates. A `region_w`-wide area around the
        click is enlarged to `inset_w` px, ruled in full-screen coordinates,
        marked, and pasted in the corner farthest from the click. Returns
        (new_image, corner_name, (x1, y1, x2, y2) shown)."""
        num, px, py = point
        w, h = img.size
        aspect = w / h
        region_h = region_w / aspect
        x1 = min(max(px - region_w / 2, 0), w - region_w)
        y1 = min(max(py - region_h / 2, 0), h - region_h)
        crop = native.crop((round(x1 * scale), round(y1 * scale),
                            round((x1 + region_w) * scale), round((y1 + region_h) * scale)))
        inset_h = round(inset_w / aspect)
        inset = crop.resize((inset_w, inset_h), Image.Resampling.LANCZOS)
        factor = inset_w / region_w
        inset = ScreenCapture.draw_click_markers(
            inset, [(num, round((px - x1) * factor), round((py - y1) * factor))])
        inset = ScreenCapture.draw_rulers(inset, origin=(x1, y1), units_per_px=1 / factor)
        # Corner farthest from the click, inside the main rulers.
        right = px < w / 2
        bottom = py < h / 2
        ox = w - inset_w - 18 if right else 18
        oy = h - inset_h - 18 if bottom else 18
        out = img.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle([ox - 3, oy - 3, ox + inset_w + 2, oy + inset_h + 2], outline=(255, 40, 40), width=3)
        out.paste(inset, (ox, oy))
        corner = ("bottom" if bottom else "top") + "-" + ("right" if right else "left")
        return out, corner, (round(x1), round(y1), round(x1 + region_w), round(y1 + region_h))

    @staticmethod
    def draw_click_markers(img: Image.Image, points: list) -> Image.Image:
        """Mark where the previous step's mouse actions landed (GCA-015):
        a red crosshair ring with the action number, so the model can see
        how far off a click was and correct its aim. `points` holds
        (number, x, y) in image coordinates. Returns a NEW image."""
        if not points:
            return img
        img = img.copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arialbd.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        red, black = (255, 40, 40), (0, 0, 0)
        for num, x, y in points:
            r = 9
            draw.ellipse([x - r, y - r, x + r, y + r], outline=black, width=3)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=red, width=2)
            draw.line([(x - r - 4, y), (x - 3, y)], fill=red, width=2)
            draw.line([(x + 3, y), (x + r + 4, y)], fill=red, width=2)
            draw.line([(x, y - r - 4), (x, y - 3)], fill=red, width=2)
            draw.line([(x, y + 3), (x, y + r + 4)], fill=red, width=2)
            label = str(num)
            draw.text((x + r + 3, y - r - 11), label, fill=black, font=font)
            draw.text((x + r + 2, y - r - 12), label, fill=red, font=font)
        return img

    @staticmethod
    def pil_to_base64(pil_img: Image.Image, quality: int = 70) -> str:
        """Convert a PIL Image to a base64-encoded JPEG string."""
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def pil_images_different(img1: Image.Image, img2: Image.Image,
                            pixel_threshold: int = 20,
                            fraction_threshold: float = 0.008) -> bool:
        """Sensitive screen change detection using pixel-level comparison.
        Resizes both images to 192x108 grayscale, counts pixels that
        changed by more than `pixel_threshold` intensity levels.
        Returns True if more than `fraction_threshold` of pixels changed.

        Tuned for card games over GFN streaming:
          pixel_threshold=20 — ignores compression noise / idle shimmer
          fraction_threshold=0.008 — 0.8% of pixels ≈ a card leaving hand
        """
        size = (192, 108)
        g1 = np.array(img1.resize(size, Image.Resampling.BILINEAR).convert("L"),
                      dtype=np.float32)
        g2 = np.array(img2.resize(size, Image.Resampling.BILINEAR).convert("L"),
                      dtype=np.float32)
        diff = np.abs(g1 - g2)
        changed = np.sum(diff > pixel_threshold)
        fraction = changed / g1.size
        return bool(fraction > fraction_threshold)

