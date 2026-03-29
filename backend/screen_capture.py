import mss
import mss.tools
import pygetwindow as gw
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64

class ScreenCapture:
    def __init__(self):
        pass  # No shared mss state — created per call for thread safety

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

    def capture(self, target_type: str, target_name: str) -> dict:
        """
        Captures the screen and returns a dict with:
          - "image": base64 encoded JPEG image
          - "offset_x": left edge of capture region in absolute screen coords
          - "offset_y": top edge of capture region in absolute screen coords
          - "scale": ratio of original_size / thumbnail_size (>=1.0)
        """
        with mss.mss() as sct:
            monitor_dict = None
            
            if target_type == "monitor":
                try:
                    mon_idx = int(target_name.replace("Monitor ", ""))
                    monitor_dict = sct.monitors[mon_idx]
                except Exception as e:
                    print(f"Error finding monitor: {e}")
                    monitor_dict = sct.monitors[1] # fallback to primary
                    
            elif target_type == "window":
                try:
                    win = gw.getWindowsWithTitle(target_name)[0]
                    # Check if it's minimized
                    if win.isMinimized:
                        win.restore()
                    # Bring to front
                    try:
                        win.activate()
                    except:
                        pass
                    monitor_dict = {
                        "top": win.top,
                        "left": win.left,
                        "width": win.width,
                        "height": win.height
                    }
                except Exception as e:
                    print(f"Error finding window: {e}")
                    monitor_dict = sct.monitors[1] # fallback
                    
            if monitor_dict is None:
                monitor_dict = sct.monitors[1]

            offset_x = monitor_dict["left"]
            offset_y = monitor_dict["top"]
                
            screenshot = sct.grab(monitor_dict)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Resize if it's too large to save tokens/bandwidth
            orig_width, orig_height = img.size
            max_size = (1920, 1080)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            thumb_width, thumb_height = img.size
            # Scale factor: how much to multiply image coords to get original coords
            scale = orig_width / thumb_width if thumb_width != orig_width else 1.0
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {
                "image": img_str,
                "pil_image": img,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "scale": scale,
            }

    def capture_fresh(self, target_type: str, target_name: str) -> Image.Image:
        """Fast capture returning only the PIL image at thumbnail size.
        Used by ActionVerifier for template matching — no base64 overhead.
        """
        with mss.mss() as sct:
            monitor_dict = None
            if target_type == "monitor":
                try:
                    mon_idx = int(target_name.replace("Monitor ", ""))
                    monitor_dict = sct.monitors[mon_idx]
                except Exception:
                    monitor_dict = sct.monitors[1]
            elif target_type == "window":
                try:
                    win = gw.getWindowsWithTitle(target_name)[0]
                    monitor_dict = {
                        "top": win.top, "left": win.left,
                        "width": win.width, "height": win.height
                    }
                except Exception:
                    monitor_dict = sct.monitors[1]
            if monitor_dict is None:
                monitor_dict = sct.monitors[1]

            screenshot = sct.grab(monitor_dict)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            max_size = (1920, 1080)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img

    def capture_phash(self, target_type: str, target_name: str) -> int:
        """Fast perceptual hash of the current screen.
        Resizes to 16x16 grayscale, thresholds against the mean,
        and returns a 256-bit integer hash. Robust against idle
        animations and minor visual noise — only meaningful changes
        (cards played, enemies dying, UI transitions) flip enough bits.
        """
        with mss.mss() as sct:
            monitor_dict = None
            if target_type == "monitor":
                try:
                    mon_idx = int(target_name.replace("Monitor ", ""))
                    monitor_dict = sct.monitors[mon_idx]
                except Exception:
                    monitor_dict = sct.monitors[1]
            elif target_type == "window":
                try:
                    win = gw.getWindowsWithTitle(target_name)[0]
                    monitor_dict = {
                        "top": win.top, "left": win.left,
                        "width": win.width, "height": win.height
                    }
                except Exception:
                    monitor_dict = sct.monitors[1]
            if monitor_dict is None:
                monitor_dict = sct.monitors[1]

            screenshot = sct.grab(monitor_dict)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
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
    def draw_rulers(img: Image.Image) -> Image.Image:
        """Draw coordinate rulers on the edges of the image.
        Helps LLMs estimate pixel positions by providing visual anchor points.
        Ticks every 100px with labels, minor ticks every 50px.
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
        major = 100  # major tick interval
        minor = 50   # minor tick interval
        major_len = 12
        minor_len = 6

        # Top ruler (horizontal)
        for x in range(0, w, minor):
            tlen = major_len if x % major == 0 else minor_len
            draw.line([(x, 0), (x, tlen)], fill=tick_color, width=1)
            if x % major == 0 and x > 0:
                txt = str(x)
                # Shadow for readability
                draw.text((x + 2, 1), txt, fill=shadow_color, font=font)
                draw.text((x + 1, 0), txt, fill=label_color, font=font)

        # Left ruler (vertical)
        for y in range(0, h, minor):
            tlen = major_len if y % major == 0 else minor_len
            draw.line([(0, y), (tlen, y)], fill=tick_color, width=1)
            if y % major == 0 and y > 0:
                txt = str(y)
                draw.text((2, y + 1), txt, fill=shadow_color, font=font)
                draw.text((1, y), txt, fill=label_color, font=font)

        # Bottom ruler (horizontal)
        for x in range(0, w, minor):
            tlen = major_len if x % major == 0 else minor_len
            draw.line([(x, h - 1), (x, h - 1 - tlen)], fill=tick_color, width=1)
            if x % major == 0 and x > 0:
                txt = str(x)
                draw.text((x + 2, h - 15), txt, fill=shadow_color, font=font)
                draw.text((x + 1, h - 16), txt, fill=label_color, font=font)

        # Right ruler (vertical)
        for y in range(0, h, minor):
            tlen = major_len if y % major == 0 else minor_len
            draw.line([(w - 1, y), (w - 1 - tlen, y)], fill=tick_color, width=1)
            if y % major == 0 and y > 0:
                txt = str(y)
                draw.text((w - 30, y + 1), txt, fill=shadow_color, font=font)
                draw.text((w - 31, y), txt, fill=label_color, font=font)

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
        return fraction > fraction_threshold

