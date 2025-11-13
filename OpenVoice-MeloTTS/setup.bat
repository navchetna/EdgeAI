@echo off
:: ======================================================
:: OpenVoice + MeloTTS Installation Script for Windows
:: ======================================================
:: Run this script as Administrator (right-click → Run as Administrator)
:: Make sure Python 3.10+ and uv are installed
:: ======================================================

echo.
echo === OpenVoice + MeloTTS Setup Script ===
echo.

:: Save starting directory (your setup folder)
set "SETUP_DIR=%cd%"

:: 1. Create and activate virtual environment
REM Check if .venv folder exists
if exist .venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    echo Creating virtual environment...
    uv venv --python=3.10
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Virtual environment ready.


echo [2/13] Activating virtual environment...
call .venv\Scripts\activate

:: 2. Install requirements
echo [3/13] Installing dependencies...
uv pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

:: 3. Install specific packages without dependencies
echo [4/13] Installing whisper-timestamped and openai-whisper...
uv pip install --no-deps "whisper-timestamped>=1.14.2"
uv pip install --no-deps openai-whisper

:: 4. Clone OpenVoice repository
echo [5/13] Cloning OpenVoice repository...
if not exist OpenVoice (
    git clone https://github.com/myshell-ai/OpenVoice.git
) else (
    echo OpenVoice folder already exists, skipping clone.
)

echo Cloning MeloTTS repository...
if not exist MeloTTS (
    git clone https://github.com/myshell-ai/MeloTTS
) else (
    echo MeloTTS folder already exists, skipping clone.
)

:: 5. Copy required project files into OpenVoice folder
echo [6/13] Copying local project files into OpenVoice folder...
set "TARGET_DIR=OpenVoice"

for %%F in (
    se_extractor_vad_fix.patch
    english.patch
    api.patch
) do (
    if exist "%SETUP_DIR%\%%F" (
        echo Copying %%F...
        copy /Y "%SETUP_DIR%\%%F" "%TARGET_DIR%\"
    ) else (
        echo WARNING: File %%F not found in setup directory, skipping.
    )
)

set "TARGET_DIR=MeloTTS"
for %%F in (
    japnese.patch
    japnese_bert.patch
    melo_english.patch
    text_init.patch
    demo_voice.wav
) do (
    if exist "%SETUP_DIR%\%%F" (
        echo Copying %%F...
        copy /Y "%SETUP_DIR%\%%F" "%TARGET_DIR%\"
    ) else (
        echo WARNING: File %%F not found in setup directory, skipping.
    )
)


:: 6. Download the model from Hugging Face
echo [7/13] Downloading OpenVoice model...
hf download myshell-ai/OpenVoice --local-dir checkpoints
hf download myshell-ai/MeloTTS-English-v3 --local-dir checkpoints/MeloTTS-English-v3

:: 6. Move into OpenVoice directory
cd OpenVoice/

echo "Current Directory: %SETUP_DIR%"

:: 7. Apply patch (if exists)
echo [8/13] Applying patch if available...
if exist se_extractor_vad_fix.patch (
    git apply se_extractor_vad_fix.patch
) else (
    echo Patch file not found, skipping patch step.
)

if exist english.patch (
    git apply english.patch
) else (
    echo Patch file not found, skipping patch step.
)

if exist api.patch (
    git apply api.patch
) else (
    echo Patch file not found, skipping patch step.
)

:: 6. Move into OpenVoice directory
cd ..\MeloTTS

if exist japnese.patch (
    git apply japnese.patch
) else (
    echo Patch file not found, skipping patch step.
)

if exist japnese_bert.patch (
    git apply japnese_bert.patch
) else (
    echo Patch file not found, skipping patch step.
)

if exist melo_english.patch (
    git apply melo_english.patch
) else (
    echo Patch file not found, skipping patch step.
)

if exist text_init.patch (
    git apply text_init.patch
) else (
    echo Patch file not found, skipping patch step.
)

cd ..\

:: 9. Quantize and compile model
echo [9/13] Running quantization...
python quantize_model.py

:: 10. Check FFmpeg availability
echo [10/13] Checking FFmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo FFmpeg not found in PATH!
    echo Please download FFmpeg from:
    echo https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z
    echo and add its bin folder to your PATH.
) else (
    ffmpeg -version
)

:: 11. Developer Mode Reminder
echo [11/13] Checking Developer Mode (required for Silero fix)...
echo Make sure Windows Developer Mode is enabled:
echo Settings → Privacy & Security → For Developers → Enable Developer Mode.

:: 12. Completion message
echo.
echo === Installation Complete! ===
echo.
echo You can now activate the environment and use OpenVoice:
echo     call "%SETUP_DIR%\.venv\Scripts\activate"
echo     cd "%SETUP_DIR%\OpenVoice"
echo     python demo.py
echo.
pause
