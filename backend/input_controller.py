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
    # 0. Pre-hover — move cursor to target so the game registers the element
    _move_to(ix, iy)
    time.sleep(0.15)
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
    time.sleep(0.08)
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

# -------------------------------------------------------------------

import json
import os

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
        """
        parts = action_string.strip().split(" ")
        cmd = parts[0].lower()
        
        try:
            if cmd == "click" and len(parts) >= 3:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                print(f"  click: img({parts[1]},{parts[2]}) -> screen({x},{y})")
                _click_at(x, y, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
            elif cmd == "right_click" and len(parts) >= 3:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                print(f"  right_click: img({parts[1]},{parts[2]}) -> screen({x},{y})")
                _click_at(x, y, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)
            elif cmd == "middle_click" and len(parts) >= 3:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                print(f"  middle_click: img({parts[1]},{parts[2]}) -> screen({x},{y})")
                _click_at(x, y, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP)
            elif cmd == "double_click" and len(parts) >= 3:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                print(f"  double_click: img({parts[1]},{parts[2]}) -> screen({x},{y})")
                _click_at(x, y, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
                time.sleep(0.05)
                _click_at(x, y, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
            elif cmd == "drag" and len(parts) >= 5:
                x1, y1 = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                x2, y2 = self._translate(int(parts[3]), int(parts[4]), offset_x, offset_y, scale_x, scale_y)
                print(f"  drag: img({parts[1]},{parts[2]})->({parts[3]},{parts[4]}) -> screen({x1},{y1})->({x2},{y2}) settle={settle_s:.2f}s")
                self._drag(x1, y1, x2, y2, settle_s=settle_s)
            elif cmd == "scroll" and len(parts) >= 4:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                amount = int(parts[3])
                print(f"  scroll: img({parts[1]},{parts[2]}) amount={amount} -> screen({x},{y})")
                _move_to(x, y)
                time.sleep(0.05)
                _send_button(MOUSEEVENTF_WHEEL, mouse_data=amount * WHEEL_DELTA)
            elif cmd == "hover" and len(parts) >= 3:
                x, y = self._translate(int(parts[1]), int(parts[2]), offset_x, offset_y, scale_x, scale_y)
                print(f"  hover: img({parts[1]},{parts[2]}) -> screen({x},{y})")
                _move_to(x, y)
            elif cmd == "press" and len(parts) >= 2:
                key = " ".join(parts[1:])
                pydirectinput.press(key)
            elif cmd == "hold_key" and len(parts) >= 2:
                key = parts[1].lower()
                pydirectinput.keyDown(key)
            elif cmd == "release_key" and len(parts) >= 2:
                key = parts[1].lower()
                pydirectinput.keyUp(key)
            elif cmd == "type" and len(parts) >= 2:
                text = " ".join(parts[1:])
                pydirectinput.write(text, interval=0.05)
            elif cmd == "wait" and len(parts) >= 2:
                sec = float(parts[1])
                time.sleep(sec)
            elif cmd == "run_macro" and len(parts) >= 2:
                macro_name = parts[1]
                self._run_macro(macro_name, offset_x, offset_y, scale_x)
            else:
                print(f"Unknown or malformed command: {action_string}")
        except Exception as e:
            print(f"Failed to execute command '{action_string}': {e}")
