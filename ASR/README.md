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



## Benchmarking
- 3 second audio
    - GPU: 5 seconds (without warmup), after warmup (2 seconds)
    - CPU: 12 seconds (without warmup), after warmup (9 seconds)
    - NPU: 