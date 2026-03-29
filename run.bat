@echo off
echo =======================================================
echo    Starting Gemini Computer Use Agent...
echo =======================================================

echo [1/2] Starting FastAPI Backend on port 8000...
start "Gemini Agent - Backend" cmd /k "cd /d c:\Coding Projects\GameCLI-Agent\backend & pip install -r requirements.txt & python -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting React Frontend on port 5173...
start "Gemini Agent - Frontend" cmd /k "cd /d c:\Coding Projects\GameCLI-Agent\frontend & npm install & npm run dev"

echo.
echo Both services are starting up in separate windows!
echo Once they are ready, you can access the UI at:
echo http://localhost:5173
echo.
pause
