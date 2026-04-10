@echo off
set PYTHONUTF8=1
set TTS_DTYPE=float32
uvicorn server:app --host 0.0.0.0 --port 8000
pause
