# PLaMo-3 (OpenVINO)

Convert and run [`pfnet/plamo-3-nict-2b-base`](https://huggingface.co/pfnet/plamo-3-nict-2b-base)
— a 2B-parameter decoder-only causal LM — on OpenVINO.

PLaMo-3 ships custom modeling code (`trust_remote_code=True`) with a
sliding-window / full-attention cache and is **not** supported by
`optimum-cli export openvino`, so it is exported directly with
`openvino.convert_model` as a **stateful (KV-cache)** model split into two graphs
that decode one token per step:

```
decoder (prefill)  input_ids [1,L]           -> logits + present.{i}.key/value (all layers)
decoder_with_past  single token + past K/V   -> logits + updated self K/V
```

Per-layer self K/V shape: `[1, 4, seq, 128]` (24 layers, 4 KV heads, head_dim 128).

## Setup
1. Create a virtual environment
    ```bash
    uv venv --python=3.10
    ```
2. Activate the environment
    - Linux/macOS: `source .venv/bin/activate`
    - Windows: `.venv\Scripts\activate`
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

This produces `ir/` containing the prefill and `decoder_with_past` graphs plus the
tokenizer and an `ov_plamo3_kvcache.json` metadata file.

## Run inference / benchmark
```bash
python inference.py --model_dir ir --device CPU
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model_dir` | `ir` | Directory with the converted IR |
| `--device` | `CPU` | OpenVINO device: `CPU` / `GPU` / `NPU` / `AUTO` |
| `--prompt` | `The capital of Japan is` | Prompt text |
| `--max_new_tokens` | `128` | Tokens generated per run |

The script prints a sample generation, then benchmarks decode throughput:
**3 warmup runs** followed by **3 timed runs**, reporting per-run and mean
**tokens/second** (early EOS is ignored during timing so each run generates the
same number of tokens for a fair comparison).

## Notes
- Correct up to the model's 4096-token `max_position_embeddings`; sliding-window
  layers self-truncate to `window_size` (2048) beyond the window.
- Greedy decoding is deterministic, so every run generates identical tokens; only
  the timing varies.
