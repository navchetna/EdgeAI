## Setup: Installation

1. Install uv:
    ```bash
    pip install uv
    ```
2. Create the environment and activate
    ```bash
    uv venv --python=3.10 
    source .venv/bin/activate     
    ```
3. Install the dependencies:
    ```bash
    uv pip install torch==2.8.0 torchaudio==2.8.0 --torch-backend=cpu
    uv pip install -r requirements.txt
    uv pip install openvino==2025.4.1
    uv pip install fastapi uvicorn
    ```

4. Clone the repository:
    ```bash
    git clone https://github.com/smtiitm/Fastspeech2_HS.git -b New-Models
    ```
    ```
5. Copy all the files from current directory to the cloned directory:
    ```
    utilities.py
    main_ov.py
    start_server.bat
    server.py
    text_preprocess_for_inference.py
    test_tts.py
    ```

## Run
1. Go to the cloned Fastspeeh2_HS directory
    `cd Fastspeech2_HS`
2. Run the main script
    `python main_ov.py` for testing
    `python start_server.py' for starting the api server
    'python test_tts.py' for testing the api server