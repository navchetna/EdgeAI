# OpenVoice 


## Installation

1. Open Command Prompt
2. Run setup.bat
    `setup.bat`
3. Enter the cloned directory:
    `cd OpenVoice`
4. Run the backend
    `python main.py`

## Testing
1. Install openai
    `uv pip install openai`
2. Run the client.py
    `python client.py`


## NOTES

- Issue 1: Install ffmpeg
-> Ensure you have installed ffmpeg in your machine. Follow this tutorial to install ffmpeg in windows
https://transloadit.com/devtips/how-to-install-ffmpeg-on-windows-a-complete-guide/



- Issue 2: 
```
OSError: [WinError 1314] A required privilege is not held by the client: 'C:\\Users\\akulshre/.cache/torch/hub/snakers4_silero-vad_v3.0' -> 'C:\\Users\\akulshre/.cache/torch/hub/snakers4_silero-vad_master'
```
-> Enable Developer Mode in your windows
