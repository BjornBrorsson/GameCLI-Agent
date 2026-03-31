# GameCLI Agent

An autonomous AI agent that plays turn-based games by watching your screen and controlling your mouse and keyboard. It uses large language models (LLMs) with vision capabilities to analyze game screenshots, make strategic decisions, and execute actions — all in real time.

> **Windows only** — relies on Windows APIs for input simulation (`pydirectinput`, `ctypes` SendInput).

## Features

- **Vision-based gameplay** — captures screenshots, draws coordinate rulers for precision, and sends them to an LLM for analysis
- **Multiple LLM providers** — Gemini CLI (free), Gemini API, or OpenRouter (access to Claude, GPT-4o, etc.)
- **Smart action verification** — OpenCV template matching + OCR to correct coordinate drift before clicking
- **Retry & recovery** — automatic retries with coordinate nudging, mid-retry LLM consultation, and loop detection
- **Game phase detection** — OCR-based detection of combat, map, card rewards, rest sites, shops, and events
- **Tiered memory** — turn-level, combat-level, and run-level memory so the agent learns from its mistakes
- **Turn boundary enforcement** — ensures the agent ends its turn when it should
- **Post-combat reflection** — LLM analyzes what went well/poorly after each fight
- **Win/loss detection** — recognizes game-over and victory screens
- **Live dashboard** — React frontend with real-time logs, screen preview, and cost tracking

## Architecture

```
frontend/          React UI (Vite)
  src/App.jsx      Main dashboard — provider/model selection, live logs, preview

backend/           Python (FastAPI + asyncio)
  main.py          REST API + WebSocket server
  agent_loop.py    Core loop: capture → detect phase → LLM → verify → execute → reflect
  llm.py           LLM integration (Gemini CLI, Gemini API, OpenRouter)
  screen_capture.py  Screenshot capture, perceptual hashing, ruler drawing
  input_controller.py  Mouse/keyboard input via Windows APIs
  action_verifier.py   OpenCV template matching + OCR coordinate correction
  game_state.py    Phase detection, OCR state extraction, tiered memory
  logger.py        Session narration + execution logging
```

## Prerequisites

- **Windows 10/11**
- **Python 3.10+**
- **Node.js 18+**
- **Tesseract OCR** — [download installer](https://github.com/UB-Mannheim/tesseract/wiki). Install to the default path (`C:\Program Files\Tesseract-OCR\`)
- **One of the following LLM providers:**
  - [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed and authenticated (free)
  - Gemini API key from [Google AI Studio](https://aistudio.google.com/)
  - OpenRouter API key from [openrouter.ai](https://openrouter.ai/)

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/BjornBrorsson/GameCLI-Agent.git
   cd GameCLI-Agent
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

4. **Create your game instructions**

   Create a file called `Game Instructions.md` in the project root:
   ```markdown
   Win a game of Slay The Spire
   ```
   This tells the agent what its objective is. Adjust for your game.

5. **Quick start** — run both backend and frontend:
   ```bash
   run.bat
   ```
   Or start them separately:
   ```bash
   # Terminal 1 — Backend
   cd backend
   python -m uvicorn main:app --host 127.0.0.1 --port 8000

   # Terminal 2 — Frontend
   cd frontend
   npm run dev
   ```

6. **Open the dashboard** at [http://localhost:5173](http://localhost:5173)

## Usage

1. Select your **LLM provider** and enter an API key (if required)
2. Choose a **model** (e.g. `gemini-2.5-flash`)
3. Pick an **agent role** (see below)
4. Select the **screen source** — a monitor or specific window
5. Click **Start** and switch to your game

The agent will:
- Capture screenshots and analyze the game state
- Decide which actions to take (play cards, click buttons, navigate menus)
- Execute actions with vision-verified precision
- Wait for opponent turns automatically
- Reflect after combats and adapt its strategy

## Agent Roles

Roles change **how** the agent interacts with the game — same core engine, different purpose.

| Role | Purpose | Behaviour |
|------|---------|-----------|
| **Gamer** | Entertainment / content creation | Plays to win with smart strategy. Narrates its thought process naturally. |
| **Reviewer** | Game critique / evaluation | Plays normally while evaluating design, UX, mechanics, difficulty, and polish. Notes positives and negatives. |
| **QA Tester** | Bug hunting / quality assurance | Systematically explores UI, tries edge cases, stress-tests interactions, and logs anomalies with repro steps. |
| **Speedrunner** | Optimization / efficiency | Minimizes actions and time. Skips optional content, prefers keyboard shortcuts, takes efficient paths. |

Select the role from the dashboard dropdown before starting the agent. The role shapes the LLM's system prompt — no code changes needed to switch between roles.

## Configuration

| Setting | Location | Description |
|---------|----------|-------------|
| Game objective | `Game Instructions.md` | What the agent should try to accomplish |
| Agent role | Dashboard UI | How the agent approaches the game (Gamer, Reviewer, Tester, Speedrunner) |
| LLM provider | Dashboard UI | Gemini CLI, Gemini API, or OpenRouter |
| Screen source | Dashboard UI | Which monitor or window to watch |
| API key | Dashboard UI | Entered at runtime, never stored on disk |

## Logs

Session logs are saved locally (git-ignored):
- `backend/Logs/` — detailed execution logs (vision verification, retries, timing)
- `backend/Narration/` — human-readable step-by-step narration of the agent's reasoning

## Tested With

- Slay the Spire

The agent is designed to be game-agnostic for turn-based games, but has primarily been tested with Slay the Spire. Contributions for other games are welcome.

## License

[MIT](LICENSE)
