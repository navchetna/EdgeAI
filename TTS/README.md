
# OpenVINO - Kokoro TTS
This demo covers the optimization and deployment of Kokoro in openAI compatible format. 


## Setup
1. Create virtual environment
    `uv venv --python=3.10`
2. Activate the environment (Windows)
    `.venv\Scripts\activate`
3. Install the requirements
    `uv pip install -r requirements.txt`
4. Optimize and compile the Kokoro model
    `python convert_to_ov.py`
5. Run the server
    `python main.py`

## Testing

- Run the `client.py` to access the endpoint for voice generation. It is openai compatible. 

## NOTE
- Please note that the intial runs for the model execution will yield poor response time. 
- Change the `MAX_BUFFER_LENGTH` in the `tts.py` depending on the max length.