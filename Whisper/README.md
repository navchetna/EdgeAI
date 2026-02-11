# ASR Pipeline


## Setup
1. Create environment
    ```uv venv --python=3.10```

2. Activate the environment (Windows)
    ```.venv\Scripts\activate```

3. Install the requirements
    ```uv pip install -r requirements.txt```

4. Download the models
    ```hf download OpenVINO/whisper-large-v3-int4-ov --local-dir whisper_large_v3```

5. Run the server
    ```python main.py```


## Test 
- Run `client.py` to access the API Endpoint
