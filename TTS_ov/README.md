



## Installation

1. uv venv --python=3.10
2. uv pip install -r requirements.txt
3. uv pip install --no-deps "whisper-timestamped>=1.14.2"
4. uv pip install --no-deps openai-whisper
5. Clone the openvoice repository
    ```https://github.com/myshell-ai/OpenVoice.git```
6. Copy all the python scripts in the OpenVoice cloned repository
7. `cd OpenVoice`
8. Download the model
    ```huggingface-cli download myshell-ai/OpenVoice --local-dir model```
9. Download ffmpeg: https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z
10. Extract the zip file. 
11. Add the path to the system environment variable.
12. Apply the patch the data, copy the patch to the cloned repo
    ```git apply se_extractor_vad_fix.patch```
13. Run the quantization and model compilation
    ```python quantize_model.py```







- For errors like
```
OSError: [WinError 1314] A required privilege is not held by the client: 'C:\\Users\\akulshre/.cache/torch/hub/snakers4_silero-vad_v3.0' -> 'C:\\Users\\akulshre/.cache/torch/hub/snakers4_silero-vad_master'
```
- Enable Developer Mode in your windows

- Error: 
```
RuntimeError: Problem when installing silero with version v3.0. Check versions here: https://github.com/snakers4/silero-vad/wiki/Version-history-and-Available-Models
```
Recreate the environment from scratch