@echo off
set PYTHONUTF8=1
set NMT_DEVICE=GPU
call .venv\Scripts\activate
uvicorn server_indic_indic:app --host 0.0.0.0 --port 8004
pause
