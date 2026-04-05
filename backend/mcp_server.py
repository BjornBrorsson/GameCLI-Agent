from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from input_controller import InputController
from screen_capture import ScreenCapture
import game_state

# Initialize MCP Server
mcp = FastMCP("GhostOS")

# Instantiate controllers
input_controller = InputController()
screen_capture = ScreenCapture()

@mcp.tool()
def get_available_screen_sources() -> Dict[str, list[str]]:
    """Get the available monitors and windows to capture from."""
    return screen_capture.get_available_sources()

@mcp.tool()
def capture_screen(target_type: str = "monitor", target_name: str = "Monitor 1") -> Dict[str, Any]:
    """
    Captures the screen.
    target_type must be either "monitor" or "window".
    target_name should be the name of the monitor (e.g. "Monitor 1") or window title.
    Returns the base64 encoded image, and offset/scale info.
    """
    return screen_capture.capture(target_type=target_type, target_name=target_name)

@mcp.tool()
def detect_game_phase(target_type: str = "monitor", target_name: str = "Monitor 1") -> Dict[str, Any]:
    """
    Captures the screen and detects the current game phase.
    Returns the phase name and the confidence score.
    """
    pil_img = screen_capture.capture_fresh(target_type=target_type, target_name=target_name)
    phase, score = game_state.detect_phase(pil_img)
    return {"phase": phase, "confidence": score}

@mcp.tool()
def extract_screen_text(target_type: str = "monitor", target_name: str = "Monitor 1") -> str:
    """
    Captures the screen and extracts text using OCR.
    """
    pil_img = screen_capture.capture_fresh(target_type=target_type, target_name=target_name)
    text = game_state.extract_screen_text(pil_img)
    return text

@mcp.tool()
def execute_action(action_string: str, offset_x: int = 0, offset_y: int = 0, scale: float = 1.0, settle_s: float = 0.5) -> str:
    """
    Executes an action via input_controller.
    Examples of action_string:
      - "click 100 200"
      - "drag 100 200 300 400"
      - "type hello"
      - "press enter"
      - "scroll 100 200 -1"
    offset_x, offset_y, and scale should typically match the ones returned by capture_screen.
    """
    try:
        input_controller.execute_action(action_string, offset_x=offset_x, offset_y=offset_y, scale=scale, settle_s=settle_s)
        return f"Successfully executed action: {action_string}"
    except Exception as e:
        return f"Failed to execute action '{action_string}': {e}"

if __name__ == "__main__":
    mcp.run(transport='stdio')
