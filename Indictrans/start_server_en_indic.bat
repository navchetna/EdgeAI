@echo off
set PYTHONUTF8=1
set NMT_DEVICE=GPU
call .venv\Scripts\activate
uvicorn server_en_indic:app --host 0.0.0.0 --port 8003
pause
