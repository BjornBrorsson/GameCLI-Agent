from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import asyncio
import os
from datetime import datetime
from typing import Optional

try:
    import requests as _requests
except ImportError:
    _requests = None

from agent_loop import AgentLoop
from screen_capture import ScreenCapture
from macro_recorder import recorder

app = FastAPI()

# Standard FastAPI CORSMiddleware is more robust for WebSockets and general use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
agent = AgentLoop()
screen_capture = ScreenCapture()
connected_websockets = set()

# Models
class StartRequest(BaseModel):
    api_key: str
    target_type: str
    target_name: str
    model_name: str
    instructions: str
    provider: str = "gemini_cli"
    role: str = "gamer"
    use_grounding: bool = False
    grounding_model: str = ""

class InstructionUpdate(BaseModel):
    instructions: str

# Helper to broadcast logs
async def broadcast_log(message: str):
    ts = datetime.now().strftime("%H:%M:%S")
    stamped = f"[{ts}] {message}"
    print(f"BCAST: {stamped}") # Also print to backend console
    for ws in list(connected_websockets):
        try:
            await ws.send_text(stamped)
        except Exception:
            connected_websockets.discard(ws)

@app.get("/api/sources")
def get_sources():
    sources = screen_capture.get_available_sources()
    return sources

class ModelsRequest(BaseModel):
    provider: str
    api_key: str = ""

class PreviewRequest(BaseModel):
    target_type: str
    target_name: str

@app.post("/api/preview")
def get_preview(req: PreviewRequest):
    try:
        capture_result = screen_capture.capture(req.target_type, req.target_name)
        return {"image": capture_result["image"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

DEFAULT_INSTRUCTIONS = """# Generic Agent Instructions
Your goal is to use the computer to accomplish basic tasks. 
1. Observe the screen carefully.
2. Outline your next step and explain why it is necessary.
3. Be aware of the bounding box coordinates of clickable elements.
4. Output EXACT coordinates for the click or drag commands."""

@app.get("/api/instructions")
def get_instructions():
    path = os.path.join(os.path.dirname(__file__), "..", "Game Instructions.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                content = DEFAULT_INSTRUCTIONS
            return {"instructions": content}
    except FileNotFoundError:
        return {"instructions": DEFAULT_INSTRUCTIONS}

@app.post("/api/instructions")
def update_instructions(req: InstructionUpdate):
    path = os.path.join(os.path.dirname(__file__), "..", "Game Instructions.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(req.instructions)
    return {"status": "ok"}

@app.post("/api/start")
async def start_agent(req: StartRequest):
    success, msg = agent.start(
        api_key=req.api_key,
        target_type=req.target_type,
        target_name=req.target_name,
        model_name=req.model_name,
        game_instructions=req.instructions,
        emit_log=broadcast_log,
        provider=req.provider,
        role=req.role,
        use_grounding=req.use_grounding,
        grounding_model=req.grounding_model
    )
    return {"status": "success" if success else "error", "message": msg}

@app.post("/api/stop")
async def stop_agent():
    success, msg = agent.stop()
    return {"status": "success" if success else "error", "message": msg}

@app.post("/api/resume")
async def resume_agent():
    success, msg = agent.resume()
    return {"status": "success" if success else "error", "message": msg}

@app.post("/api/abort")
async def abort_agent():
    success, msg = agent.abort()
    return {"status": "success" if success else "error", "message": msg}

@app.get("/api/status")
def get_status():
    return {"is_running": agent.is_running, "is_paused": agent.is_paused, "is_recording_macro": recorder.recording}

class MacroStartRequest(BaseModel):
    macro_name: str

@app.post("/api/macros/start_recording")
async def start_macro_recording(req: MacroStartRequest):
    success, msg = recorder.start_recording(req.macro_name)
    return {"status": "success" if success else "error", "message": msg}

@app.post("/api/macros/stop_recording")
async def stop_macro_recording():
    success, msg = recorder.stop_recording()
    return {"status": "success" if success else "error", "message": msg}

@app.get("/api/macros")
def get_macros():
    return {"macros": list(recorder.get_macros().keys())}

@app.get("/api/session")
def get_session():
    """Check if a resumable session exists from a prior crash."""
    state = agent.session_state.load()
    if state and not agent.is_running:
        return {"resumable": True, "session": state}
    return {"resumable": False}

@app.post("/api/session/clear")
def clear_session():
    agent.session_state.clear()
    return {"status": "success"}

@app.get("/api/recipes")
def list_recipes():
    return {"recipes": agent.recipes.list_all()}

@app.post("/api/recipes/{index}/toggle")
def toggle_recipe(index: int, enabled: bool = True):
    ok = agent.recipes.toggle(index, enabled)
    return {"status": "success" if ok else "error"}

@app.delete("/api/recipes/{index}")
def delete_recipe(index: int):
    ok = agent.recipes.delete(index)
    return {"status": "success" if ok else "error"}

# ── Gemini CLI fallback model list (no API key needed) ──
GEMINI_CLI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]

@app.post("/api/models")
def get_models(req: ModelsRequest):
    """Fetch available models from the provider's API.
    Returns {models: [{id, name}], default: str}.
    """
    if req.provider == "gemini_cli":
        return {
            "models": [{"id": m, "name": m} for m in GEMINI_CLI_MODELS],
            "default": "gemini-2.5-flash",
        }

    if _requests is None:
        return JSONResponse(status_code=500, content={"error": "requests package not installed"})

    try:
        if req.provider == "gemini_api":
            return _fetch_gemini_models(req.api_key)
        elif req.provider == "openrouter":
            return _fetch_openrouter_models(req.api_key)
        else:
            return JSONResponse(status_code=400, content={"error": f"Unknown provider: {req.provider}"})
    except _requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        if status in (400, 401, 403):
            msg = "Invalid API key. Please check and try again."
        elif status == 429:
            msg = "Rate limited. Please wait a moment and try again."
        else:
            msg = f"Provider returned HTTP {status}"
        print(f"[models] {req.provider} HTTP {status}: {e}")
        return JSONResponse(status_code=400, content={"error": msg})
    except _requests.exceptions.Timeout:
        return JSONResponse(status_code=504, content={"error": "Request timed out. Try again."})
    except Exception as e:
        print(f"[models] Error fetching models for {req.provider}: {e}")
        return JSONResponse(status_code=502, content={"error": str(e)})


def _fetch_gemini_models(api_key: str):
    """List models from Gemini API, filtered to generative vision models."""
    resp = _requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    models = []
    for m in data.get("models", []):
        name = m.get("name", "")  # e.g. "models/gemini-2.5-flash"
        model_id = name.replace("models/", "")
        # Only include models that support vision (generateContent)
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue
        display = m.get("displayName", model_id)
        models.append({"id": model_id, "name": display})
    # Sort: preview/latest first, then alphabetical
    models.sort(key=lambda x: (0 if "flash" in x["id"] else 1, x["id"]))
    default = "gemini-2.5-flash" if any(m["id"] == "gemini-2.5-flash" for m in models) else (models[0]["id"] if models else "")
    return {"models": models, "default": default}


def _fetch_openrouter_models(api_key: str):
    """List models from OpenRouter, filtered to vision-capable models."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = _requests.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    models = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        display = m.get("name", model_id)
        # Include models that support images
        modality = m.get("architecture", {}).get("modality", "")
        if "image" not in modality and "vision" not in modality and "multimodal" not in modality:
            # Also check top_provider or description for vision hints
            desc = (m.get("description", "") or "").lower()
            if "vision" not in desc and "image" not in desc and "multimodal" not in desc:
                continue
        # Add pricing info to display name
        pricing = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0)) * 1_000_000  # per 1M tokens
        completion_price = float(pricing.get("completion", 0)) * 1_000_000
        if prompt_price > 0:
            display = f"{display}  (${prompt_price:.2f}/${completion_price:.2f} per 1M tok)"
        models.append({"id": model_id, "name": display})
    # Sort by provider grouping
    models.sort(key=lambda x: x["id"])
    default = "google/gemini-2.5-flash-preview" if any(m["id"] == "google/gemini-2.5-flash-preview" for m in models) else (models[0]["id"] if models else "")
    return {"models": models, "default": default}


@app.get("/api/cost")
def get_cost():
    """Return session cost summary (tokens + USD) for API providers."""
    if agent.llm and hasattr(agent.llm, 'cost'):
        return agent.llm.cost.get_summary()
    return {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0, "call_count": 0}

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            # Keep connection open, wait for client drop
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.discard(websocket)
