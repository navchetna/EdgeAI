# IndicWav2vec Model Optimization
This directory contains scripts and instructions to optimize the IndicWav2vec ASR models for better performance and efficiency.


## Setup: Installation
1. Create a virtual environment and activate it:
    ```bash
    uv venv --python=3.10
    .venv\Scripts\activate
    ```
2. Install the dependencies:
    ```bash
    uv pip install -r requirements.txt
    uv pip install torch==2.8.0 --torch-backend=cpu
    uv pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.1
    ```
2. Download the model
    ```bash
    curl -L -o hindi.pt https://asr.iitm.ac.in/SPRING_INX/models/fine_tuned/SPRING_INX_ccc_wav2vec2_Hindi.pt
    ```
3. Run the script to download the model and test audio:
    ```bash
    python main.py
    ```

## Run the script

1. Activate the environment
    ```bash
    source .venv/bin/activate
    ```
2. Run the main script
    ```bash
    python main.py
    ```

### OV

1. Run the script wav2vec_torchsript.py:
    ```bash
    python wav2vec_ov.py
    ```


### Update
1. Wav2vec functional on GPU and CPU. 
2. Performance:
    10 seconds audio
    CPU - 3.4-3.8 seconds
    GPU - 0.5 seconds