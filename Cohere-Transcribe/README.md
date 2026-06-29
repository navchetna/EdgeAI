# Cohere Transcribe (OpenVINO)

Convert and run [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
— a 2B-parameter conformer encoder / transformer decoder ASR model — on OpenVINO.

The exported model is **stateful (KV-cache)**: it is split into three graphs and
decodes one token per step, reusing cached self- and cross-attention key/values.

```
encoder            input_features [1,T,128] -> encoder_hidden_states [1,T',1280]
decoder (prefill)  decoder_input_ids        -> logits + self/cross K/V (all layers)
decoder_with_past  single token + past K/V  -> logits + updated self K/V
```

## Setup
1. Create a virtual environment
    ```bash
    uv venv --python=3.10
    ```
2. Activate the environment (Windows)
    ```bash
    .venv\Scripts\activate
    ```
3. Install the requirements
    ```bash
    uv pip install -r requirements.txt
    ```

## Convert the model
Export the OpenVINO IR (downloads ~4 GB on first run):

```bash
python convert_to_ov.py --output_dir ir --weight_format fp16
```

| Flag | Default | Description |
| --- | --- | --- |
| `--output_dir` | `ir` | Directory for the OpenVINO IR files |
| `--weight_format` | `fp16` | `fp16` (half size, recommended) or `fp32` |

This produces `ir/` containing the encoder, decoder, and `decoder_with_past`
graphs plus the processor and an `ov_cohere_transcribe_kvcache.json` metadata file.

## Run inference
```bash
python inference.py --model_dir ir --device CPU --audio sample.wav
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model_dir` | `ir` | Directory with the converted IR |
| `--device` | `CPU` | OpenVINO device: `CPU` / `GPU` / `NPU` / `AUTO` |
| `--audio` | _(demo clip)_ | Path to a 16 kHz mono `.wav`; if omitted a LibriSpeech sample is downloaded |
| `--max_new_tokens` | `256` | Maximum decoded tokens |

The script prints the transcription and end-to-end latency.

## Notes
- Audio is resampled to 16 kHz mono internally.
- Cross-attention K/V are computed once at prefill and reused every step; self
  K/V grow by one token per step, which keeps the decode loop lightweight.
