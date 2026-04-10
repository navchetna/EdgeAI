@echo off
set PYTHONUTF8=1
set ASR_DEVICE=GPU
set ASR_MODEL_DIR=openvino_models
set ASR_CONFIG_PATH=.
call .venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8002
pause
