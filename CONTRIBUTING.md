# Contributing to GameCLI Agent

Thanks for your interest in contributing! Here's how to get started.

## Getting Started

1. Fork the repository
2. Clone your fork and follow the [setup instructions](README.md#setup)
3. Create a feature branch: `git checkout -b my-feature`
4. Make your changes
5. Test with an actual game session to verify behavior
6. Commit and push: `git push origin my-feature`
7. Open a Pull Request

## Development Notes

- **Windows only** — the agent uses `pydirectinput`, `ctypes` SendInput, and `pygetwindow`, which are Windows-specific
- **Tesseract OCR** must be installed for game state extraction and action verification
- The backend runs on **FastAPI** with asyncio; the agent loop is fully async
- The frontend is a **React + Vite** app communicating via REST and WebSocket

## Project Structure

```
backend/
  agent_loop.py       Main loop — the core of the agent
  llm.py              LLM provider integration and prompts
  game_state.py       Phase detection, OCR extraction, tiered memory
  action_verifier.py  OpenCV template matching + OCR verification
  input_controller.py Mouse/keyboard input
  screen_capture.py   Screenshot capture and processing
  logger.py           Session and execution logging
  main.py             FastAPI server

frontend/
  src/App.jsx         Dashboard UI
```

## Areas for Contribution

- **Game support** — test with new turn-based games and document any needed adjustments
- **Cross-platform** — help port input/capture to Linux or macOS
- **OCR accuracy** — improve game state extraction for different resolutions and games
- **Phase detection** — add keyword sets or heuristics for games beyond Slay the Spire
- **LLM prompts** — refine system prompts for better strategic decision-making
- **Tests** — add unit and integration tests

## Code Style

- Follow existing patterns and naming conventions
- Keep changes focused — one feature or fix per PR
- Don't add or remove comments unless that's the purpose of the change
- Run `python -m py_compile <file>` to verify syntax before committing

## Reporting Issues

When filing a bug report, please include:
- Steps to reproduce
- Which game you were playing
- Relevant logs from `backend/Logs/` (sanitize any personal info)
- Your Python version and OS version
