# OpenVoice 


## Installation

1. Open Command Prompt
2. Run setup.bat

## Testing
1. Install openai
    `uv pip install openai`
2. Run the client.py
    `python client.py`








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