import pydirectinput
import ctypes
import ctypes.wintypes
import time

# --- Ensure DPI awareness BEFORE any coordinate queries ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

# --- SendInput constants ---
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE       = 0x0001
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_RIGHTDOWN  = 0x0008
MOUSEEVENTF_RIGHTUP    = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP   = 0x0040
MOUSEEVENTF_WHEEL      = 0x0800
MOUSEEVENTF_ABSOLUTE   = 0x8000
WHEEL_DELTA = 120

class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput)]

class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("iu", _InputUnion)]

_screen_w = ctypes.windll.user32.GetSystemMetrics(0)
_screen_h = ctypes.windll.user32.GetSystemMetrics(1)
print(f"[InputController] Screen metrics: {_screen_w}x{_screen_h} (DPI-aware)")

def _send_button(flags, mouse_data=0):
    """Send a button-only mouse event (no position, no ABSOLUTE).
    GFN and other streaming clients read the OS cursor position via
    GetCursorPos, so we position with SetCursorPos first, then fire
    a bare button event. This avoids MOUSEEVENTF_ABSOLUTE which
    streaming clients may handle differently.
    """
    extra = ctypes.c_ulong(0)
    mi = _MouseInput(0, 0, mouse_data, flags, 0, ctypes.pointer(extra))
    iu = _InputUnion(mi=mi)
    inp = _Input(type=INPUT_MOUSE, iu=iu)
    result = ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))
    if result == 0:
        err = ctypes.get_last_error()
        print(f"  [!] SendInput FAILED flags=0x{flags:04x} error={err}")
    return result

def _move_to(x, y):
    """Position cursor using BOTH methods for maximum compatibility:
    1. SetCursorPos — GFN / streaming clients read this via GetCursorPos
    2. SendInput with MOUSEEVENTF_MOVE|ABSOLUTE — local / DirectInput games
       need this because they don't poll GetCursorPos.
    """
    ix, iy = int(x), int(y)
    # Method 1: OS cursor (GFN reads this)
    ctypes.windll.user32.SetCursorPos(ix, iy)
    # Method 2: SendInput absolute move (local/DirectInput games need this)
    abs_x = int(ix * 65536 / _screen_w) + 1
    abs_y = int(iy * 65536 / _screen_h) + 1
    extra = ctypes.c_ulong(0)
    mi = _MouseInput(abs_x, abs_y, 0,
                     MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                     0, ctypes.pointer(extra))
    iu = _InputUnion(mi=mi)
    inp = _Input(type=INPUT_MOUSE, iu=iu)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))
    time.sleep(0.01)

def _click_at(x, y, down_flag, up_flag):
    """Click at (x, y) using combined position+button events for max compatibility.
    Each button event carries MOUSEEVENTF_ABSOLUTE coordinates so games that
    don't correlate separate MOVE and BUTTON events still see the position.
    Also does SetCursorPos first for GFN / streaming clients.
    Pre-hovers at the target for 150ms so the game registers the cursor
    before the click — many UIs need this to activate hover states / hitboxes.
    """
    ix, iy = int(x), int(y)
    # 0. Pre-hover — move cursor to target so the game registers the element.
    # Streaming clients relay input with extra latency, so hover/hold longer.
    _move_to(ix, iy)
    time.sleep(0.3 if _streaming else 0.15)
    # 1. SetCursorPos for GFN (reads via GetCursorPos)
    ctypes.windll.user32.SetCursorPos(ix, iy)
    time.sleep(0.05)
    # 2. Normalized absolute coords for SendInput
    abs_x = int(ix * 65536 / _screen_w) + 1
    abs_y = int(iy * 65536 / _screen_h) + 1
    extra = ctypes.c_ulong(0)
    # 3. Button DOWN with absolute position
    mi_down = _MouseInput(abs_x, abs_y, 0,
                          down_flag | MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                          0, ctypes.pointer(extra))
    iu_down = _InputUnion(mi=mi_down)
    inp_down = _Input(type=INPUT_MOUSE, iu=iu_down)
    result = ctypes.windll.user32.SendInput(1, ctypes.pointer(inp_down), ctypes.sizeof(inp_down))
    if result == 0:
        err = ctypes.get_last_error()
        print(f"  [!] SendInput click-down FAILED error={err}")
    time.sleep(0.15 if _streaming else 0.08)
    # 4. Button UP with absolute position
    extra2 = ctypes.c_ulong(0)
    mi_up = _MouseInput(abs_x, abs_y, 0,
                        up_flag | MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                        0, ctypes.pointer(extra2))
    iu_up = _InputUnion(mi=mi_up)
    inp_up = _Input(type=INPUT_MOUSE, iu=iu_up)
    result = ctypes.windll.user32.SendInput(1, ctypes.pointer(inp_up), ctypes.sizeof(inp_up))
    if result == 0:
        err = ctypes.get_last_error()
        print(f"  [!] SendInput click-up FAILED error={err}")
        if _target_hwnd is not None and down_flag == MOUSEEVENTF_LEFTDOWN:
            pt = ctypes.wintypes.POINT(ix, iy)
            ctypes.windll.user32.ScreenToClient(_target_hwnd, ctypes.byref(pt))
            lparam = ((pt.y & 0xFFFF) << 16) | (pt.x & 0xFFFF)
            ctypes.windll.user32.PostMessageW(_target_hwnd, WM_MOUSEMOVE, 0, lparam)
            time.sleep(0.04)
            ctypes.windll.user32.PostMessageW(_target_hwnd, WM_LBUTTONDOWN, 1, lparam)
            time.sleep(0.08)
            ctypes.windll.user32.PostMessageW(_target_hwnd, WM_LBUTTONUP, 0, lparam)
            print(f"  [*] Fallback PostMessage click to client ({pt.x},{pt.y}) on hwnd={_target_hwnd}")

# -------------------------------------------------------------------
# Target-window safety: never send input to a window other than the game.

import json
import os

WM_MOUSEMOVE   = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP   = 0x0202
WM_KEYDOWN     = 0x0100
WM_KEYUP       = 0x0101
GA_ROOT = 2

_target_hwnd = None
_streaming = False  # target is a cloud-streaming client (GeForce NOW etc.)


def set_streaming(on: bool):
    """Streaming clients need longer hover/hold times, and keys posted to
    their window never reach the remote game (GCA-017)."""
    global _streaming
    _streaming = bool(on)


class InputBlockedError(RuntimeError):
    """The input would land on a window other than the target game."""


def set_target_window(hwnd, left=0, top=0):
    """Register the game window. While set, mouse input is only sent to points
    inside it and keyboard input only while it is in the foreground."""
    global _target_hwnd
    _target_hwnd = hwnd


def _root_window(hwnd):
    return ctypes.windll.user32.GetAncestor(hwnd, GA_ROOT) if hwnd else 0


def _window_title(hwnd) -> str:
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value


def _bring_to_front(hwnd):
    """Best-effort foreground switch (Alt trick lets SetForegroundWindow work
    from a background process)."""
    user32 = ctypes.windll.user32
    user32.keybd_event(0x12, 0, 0, 0)  # Alt down
    user32.SetForegroundWindow(hwnd)
    user32.keybd_event(0x12, 0, 2, 0)  # Alt up
    time.sleep(0.15)


def _ensure_point_on_target(x: int, y: int):
    """Raise InputBlockedError unless screen point (x, y) shows the target
    window. Tries one foreground switch first."""
    if _target_hwnd is None:
        return
    user32 = ctypes.windll.user32
    for attempt in range(2):
        hit = _root_window(user32.WindowFromPoint(ctypes.wintypes.POINT(int(x), int(y))))
        if hit == _root_window(_target_hwnd):
            return
        if attempt == 0:
            _bring_to_front(_target_hwnd)
    raise InputBlockedError(
        f"point ({int(x)},{int(y)}) is covered by '{_window_title(hit) or hit}', not the game window")


def _target_is_foreground() -> bool:
    if _target_hwnd is None:
        return True
    fg = _root_window(ctypes.windll.user32.GetForegroundWindow())
    if fg == _root_window(_target_hwnd):
        return True
    _bring_to_front(_target_hwnd)
    return _root_window(ctypes.windll.user32.GetForegroundWindow()) == _root_window(_target_hwnd)


# Virtual-key codes for the PostMessage fallback (game not in the foreground).
_VK_NAMED = {
    "space": 0x20, "enter": 0x0D, "return": 0x0D, "escape": 0x1B, "esc": 0x1B,
    "tab": 0x09, "backspace": 0x08, "delete": 0x2E, "insert": 0x2D,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
}


def _vk_for(key: str) -> int:
    if key in _VK_NAMED:
        return _VK_NAMED[key]
    if len(key) == 1 and key.isalnum():
        return ord(key.upper())
    return 0


def _post_key(key: str) -> bool:
    vk = _vk_for(key)
    if not vk or _target_hwnd is None:
        return False
    ctypes.windll.user32.PostMessageW(_target_hwnd, WM_KEYDOWN, vk, 0)
    time.sleep(0.05)
    ctypes.windll.user32.PostMessageW(_target_hwnd, WM_KEYUP, vk, 0xC0000001)
    return True

MACROS_FILE = os.path.join(os.path.dirname(__file__), "..", "macros.json")

class InputController:
    def __init__(self):
        pydirectinput.PAUSE = 0.02
        
    def _run_macro(self, macro_name: str, offset_x: int, offset_y: int, scale_x: float = 1.0):
        """Loads and executes a saved macro sequence."""
        if not os.path.exists(MACROS_FILE):
            print(f"  [!] Macro file not found: {MACROS_FILE}")
            return
        try:
            with open(MACROS_FILE, "r", encoding="utf-8") as f:
                macros = json.load(f)
        except json.JSONDecodeError:
            print("  [!] Failed to parse macros.json")
            return
        if macro_name not in macros:
            print(f"  [!] Macro not found: {macro_name}")
            return
        events = macros[macro_name]
        print(f"  [macro] Executing '{macro_name}' ({len(events)} events)")
        for event in events:
            delay = event.get("delay", 0)
            if delay > 0:
                time.sleep(delay)
            e_type = event.get("type")
            action = event.get("action")
            if e_type == "mouse_click":
                x, y = event.get("x", 0), event.get("y", 0)
                btn = event.get("button", "left").replace("Button.", "")
                flags_down = MOUSEEVENTF_LEFTDOWN
                flags_up = MOUSEEVENTF_LEFTUP
                if btn == "right":
                    flags_down = MOUSEEVENTF_RIGHTDOWN
                    flags_up = MOUSEEVENTF_RIGHTUP
                elif btn == "middle":
                    flags_down = MOUSEEVENTF_MIDDLEDOWN
                    flags_up = MOUSEEVENTF_MIDDLEUP
                _move_to(x, y)
                if action == "down":
                    _send_button(flags_down)
                elif action == "up":
                    _send_button(flags_up)
            elif e_type == "mouse_scroll":
                x, y = event.get("x", 0), event.get("y", 0)
                dy = event.get("dy", 0)
                _move_to(x, y)
                _send_button(MOUSEEVENTF_WHEEL, mouse_data=int(dy * WHEEL_DELTA))
            elif e_type == "key_press":
                key = event.get("key", "")
                if action == "down":
                    pydirectinput.keyDown(key)
                elif action == "up":
                    pydirectinput.keyUp(key)

    def _translate(self, x: int, y: int, offset_x: int, offset_y: int,
                   scale_x: float = 1.0, scale_y: float = 1.0) -> tuple:
        """Translate image-space coords to absolute screen coords."""
        abs_x = int(x * scale_x) + offset_x
        abs_y = int(y * scale_y) + offset_y
        return abs_x, abs_y

    def _drag(self, x1, y1, x2, y2, settle_s: float = 0.5):
        """
        Perform a drag using SetCursorPos for all movement.
        Timings are tuned for GFN / streaming clients where each
        input has ~30-80ms of round-trip latency before the cloud
        game sees it.
        settle_s: how long to hold at the destination before releasing.
                  Increased on retries to give GFN more time to relay.
        """
        # 1. Position cursor at start (card location)
        _move_to(x1, y1)
        time.sleep(0.15)

        # 2. Press button — card pickup
        _send_button(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.3)   # GFN must relay click; cloud game must register card pickup

        # 3. Interpolate movement — drag card toward target
        steps = 20
        for i in range(1, steps + 1):
            ix = int(x1 + (x2 - x1) * i / steps)
            iy = int(y1 + (y2 - y1) * i / steps)
            _move_to(ix, iy)
            time.sleep(0.025)  # ~500ms total drag movement

        # 4. Settle at destination — targeting arrow must lock on through GFN latency
        _move_to(int(x2), int(y2))
        time.sleep(settle_s)

        # 5. Release button — play the card
        _send_button(MOUSEEVENTF_LEFTUP)

    def execute_action(self, action_string: str, offset_x: int = 0, offset_y: int = 0,
                       scale_x: float = 1.0, scale_y: float = 1.0, settle_s: float = 0.5):
        """
        Executes a single command parsed from the LLM.
        All mouse operations use SendInput + SetCursorPos for consistency.
        Keyboard operations use pydirectinput.
        settle_s: drag settle time at destination (increased on retries).

        Returns "ok", "blocked: <why>" when the input would have landed on
        another window (nothing was sent), or "error: <why>".
        """
        parts = action_string.strip().split(" ")
        cmd = parts[0].lower()

        try:
            if cmd in ("click", "right_click", "middle_click", "double_click") and len(parts) >= 3:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                _ensure_point_on_target(x, y)
                print(f"  {cmd}: img({parts[1]},{parts[2]}) -> screen({x},{y})")
                down, up = {"click": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
                            "double_click": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
                            "right_click": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
                            "middle_click": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP)}[cmd]
                _click_at(x, y, down, up)
                if cmd == "double_click":
                    time.sleep(0.05)
                    _click_at(x, y, down, up)
            elif cmd == "drag" and len(parts) >= 5:
                x1, y1 = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                x2, y2 = self._translate(int(parts[3]), int(parts[4]), offset_x, offset_y, scale_x, scale_y)
                _ensure_point_on_target(x1, y1)
                _ensure_point_on_target(x2, y2)
                print(f"  drag: img({parts[1]},{parts[2]})->({parts[3]},{parts[4]}) -> screen({x1},{y1})->({x2},{y2}) settle={settle_s:.2f}s")
                self._drag(x1, y1, x2, y2, settle_s=settle_s)
            elif cmd == "scroll" and len(parts) >= 4:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                _ensure_point_on_target(x, y)
                amount = int(parts[3])
                print(f"  scroll: img({parts[1]},{parts[2]}) amount={amount} -> screen({x},{y})")
                _move_to(x, y)
                time.sleep(0.05)
                _send_button(MOUSEEVENTF_WHEEL, mouse_data=amount * WHEEL_DELTA)
            elif cmd == "hover" and len(parts) >= 3:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                _ensure_point_on_target(x, y)
                print(f"  hover: img({parts[1]},{parts[2]}) -> screen({x},{y})")
                _move_to(x, y)
            elif cmd in ("press", "hold_key", "release_key", "type") and len(parts) >= 2:
                # Keys go to whatever window has focus, so only send them
                # while the game is in front. Otherwise post single key
                # presses straight to the game window (never both).
                key = " ".join(parts[1:]).lower() if cmd == "press" else parts[1].lower()
                if not _target_is_foreground():
                    if cmd == "press" and not _streaming and _post_key(key):
                        print(f"  press {key}: game not in front — posted to its window")
                        return "ok"
                    raise InputBlockedError("game window is not in the foreground")
                if cmd == "press":
                    pydirectinput.press(key)
                elif cmd == "hold_key":
                    pydirectinput.keyDown(key)
                elif cmd == "release_key":
                    pydirectinput.keyUp(key)
                else:
                    pydirectinput.write(" ".join(parts[1:]), interval=0.05)
            elif cmd == "wait" and len(parts) >= 2:
                sec = float(parts[1])
                time.sleep(sec)
            elif cmd == "run_macro" and len(parts) >= 2:
                macro_name = parts[1]
                self._run_macro(macro_name, offset_x, offset_y, scale_x)
            else:
                print(f"Unknown or malformed command: {action_string}")
                return "error: unknown or malformed command"
        except InputBlockedError as e:
            print(f"  [safety] Input blocked — {e}")
            return f"blocked: {e}"
        except Exception as e:
            print(f"Failed to execute command '{action_string}': {e}")
            return f"error: {e}"
        return "ok"
