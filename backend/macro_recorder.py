import json
import os
import time
import threading
from pynput import mouse, keyboard

MACROS_FILE = os.path.join(os.path.dirname(__file__), "..", "macros.json")

class MacroRecorder:
    def __init__(self):
        self.recording = False
        self.events = []
        self.start_time = 0
        self.mouse_listener = None
        self.keyboard_listener = None
        self.current_macro_name = ""

    def start_recording(self, macro_name: str):
        if self.recording:
            return False, "Already recording."

        self.recording = True
        self.events = []
        self.start_time = time.time()
        self.current_macro_name = macro_name

        self.mouse_listener = mouse.Listener(
            on_click=self.on_click,
            on_scroll=self.on_scroll
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.mouse_listener.start()
        self.keyboard_listener.start()
        print(f"Started recording macro: {macro_name}")
        return True, f"Started recording {macro_name}."

    def stop_recording(self):
        if not self.recording:
            return False, "Not recording."

        self.recording = False
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        self._save_macro()
        print(f"Stopped recording macro: {self.current_macro_name}")
        return True, f"Stopped and saved macro: {self.current_macro_name}"

    def _save_macro(self):
        if not self.events:
            print("No events to save.")
            return

        macros = self.get_macros()
        macros[self.current_macro_name] = self.events

        with open(MACROS_FILE, "w", encoding="utf-8") as f:
            json.dump(macros, f, indent=2)

    def get_macros(self):
        if not os.path.exists(MACROS_FILE):
            return {}
        try:
            with open(MACROS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _add_event(self, event_type, **kwargs):
        if not self.recording:
            return

        # Calculate delay since last event or start of recording
        t = time.time()
        delay = t - self.start_time
        if self.events:
            delay = t - self.events[-1]["time"]

        event = {
            "type": event_type,
            "delay": round(delay, 3),
            "time": t
        }
        event.update(kwargs)
        self.events.append(event)

    def on_click(self, x, y, button, pressed):
        if pressed:
            # We record x,y, but during execution we might need to relate this to the game window
            # For now, recording raw coordinates.
            self._add_event("mouse_click", x=x, y=y, button=button.name, action="down")
        else:
            self._add_event("mouse_click", x=x, y=y, button=button.name, action="up")

    def on_scroll(self, x, y, dx, dy):
        self._add_event("mouse_scroll", x=x, y=y, dx=dx, dy=dy)

    def on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key.name
        self._add_event("key_press", key=k, action="down")

    def on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key.name
        self._add_event("key_press", key=k, action="up")

# Global instance for FastAPI to use
recorder = MacroRecorder()
