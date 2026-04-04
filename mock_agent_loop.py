import sys
from unittest.mock import MagicMock

# Create a mock for mss
class MockMss:
    class tools:
        pass
sys.modules['mss'] = MockMss()
sys.modules['mss.tools'] = MagicMock()
sys.modules['pydirectinput'] = MagicMock()
sys.modules['ctypes.wintypes'] = MagicMock()
sys.modules['pygetwindow'] = MagicMock()

class MockWindll:
    class user32:
        SendInput = MagicMock()
        GetSystemMetrics = MagicMock(return_value=1000)
    class shcore:
        SetProcessDpiAwareness = MagicMock()

import ctypes
ctypes.windll = MockWindll()

from backend.agent_loop import AgentLoop
id_map = {1: (100, 200), 2: (300, 400), 5: (500, 600)}

print(AgentLoop._parse_action({"command": "click", "target_id": 1, "reason": "Test click"}, id_map))
print(AgentLoop._parse_action({"command": "drag", "source_id": 2, "target_id": 5, "reason": "Test drag"}, id_map))
print(AgentLoop._parse_action({"command": "scroll", "target_id": 1, "amount": 10, "reason": "Test scroll"}, id_map))
print(AgentLoop._parse_action({"command": "hover", "target_id": 5, "reason": "Test hover"}, id_map))
print(AgentLoop._parse_action("press enter", id_map))
