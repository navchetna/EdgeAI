# Supertonic TTS - OpenVINO Edition

Supertonic is a high-quality multilingual Text-to-Speech (TTS) system. This repository provides tools to convert ONNX models to OpenVINO IR format and run inference on CPU or GPU with optimized performance.

## Features

- **Multilingual Support**: 30+ languages including English, Hindi, Korean, Japanese, and more
- **OpenVINO Acceleration**: Optimized inference on Intel CPUs and GPUs
- **Flexible Precision**: FP16 (faster, smaller) or FP32 (maximum accuracy)
- **Voice Styles**: Customizable voice characteristics via style embeddings
- **Batch Processing**: Efficient batch inference for multiple texts

## Project Structure

```
Supertonic/
├── convert_to_ov.py    # ONNX to OpenVINO converter
├── inference.py        # OpenVINO inference script
├── helper.py           # Utility functions and ONNX runtime support
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── supertonic-3/       # Auto-cloned model repository
    ├── onnx/           # ONNX model weights
    │   ├── duration_predictor.onnx
    │   ├── text_encoder.onnx
    │   ├── vector_estimator.onnx
    │   ├── vocoder.onnx
    │   ├── tts.json
    │   └── unicode_indexer.json
    └── voice_styles/   # Voice style embeddings
        └── *.json
```

## Environment Setup

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify OpenVINO Installation

```bash
python -c "from openvino.runtime import Core; print('OpenVINO:', Core().available_devices)"
```

Expected output shows available devices: `['CPU']` or `['CPU', 'GPU']`

### 4. GPU Support (Optional)

For Intel GPU acceleration, install the GPU drivers and OpenVINO GPU plugin:

```bash
# Ubuntu/Debian
sudo apt-get install intel-opencl-icd

# Verify GPU is detected
python -c "from openvino.runtime import Core; c=Core(); print('GPU' in c.available_devices)"
```

## Model Conversion

Convert ONNX models to OpenVINO IR format using `convert_to_ov.py`.

**Note:** The script will automatically clone the Supertonic model repository from Hugging Face if the `supertonic-3` directory doesn't exist.

### Basic Usage

```bash
# Convert to FP16 (default) - auto-downloads models if needed
python convert_to_ov.py

# Convert to FP32 (maximum accuracy)
python convert_to_ov.py --precision fp32
```

### Command Line Options

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--repo-dir` | `-r` | `supertonic-3` | Repository directory (auto-clones if missing) |
| `--input-dir` | `-i` | `<repo-dir>/onnx` | Directory containing ONNX models |
| `--output-dir` | `-o` | `ov_model_{precision}` | Output directory for OpenVINO models |
| `--precision` | `-p` | `fp16` | Model precision: `fp16` or `fp32` |
| `--device` | `-d` | `CPU` | Target device: `CPU` or `GPU` |

### Examples

```bash
# Default: auto-clone repo, convert to FP16
python convert_to_ov.py

# Custom output directory
python convert_to_ov.py -o my_ov_models -p fp16

# FP32 for GPU (best accuracy)
python convert_to_ov.py -p fp32 -d GPU

# Use a different repo location
python convert_to_ov.py -r /path/to/supertonic-3

# View help
python convert_to_ov.py --help
```

### Expected Output

```
Repository not found at 'supertonic-3'. Cloning from Hugging Face...
  URL: https://huggingface.co/Supertone/supertonic-3

Repository cloned successfully to 'supertonic-3'

======================================================================
Converting Supertonic TTS ONNX Models to OpenVINO IR Format
======================================================================

  Input directory:  supertonic-3/onnx
  Output directory: ov_model_fp16
  Precision:        FP16
  Target device:    CPU

----------------------------------------------------------------------
  Converting duration_predictor (FP16)...
  OK   - duration_predictor (0.52MB + 1.24MB)
  Converting text_encoder (FP16)...
  OK   - text_encoder (2.31MB + 45.67MB)
  ...
======================================================================
```

## Inference

Run TTS inference using the converted OpenVINO models.

**Note:** The script will automatically clone the Supertonic model repository if needed, and use default voice styles from `supertonic-3/voice_styles/`.

### Basic Usage

```bash
# Minimal - uses defaults for model dir and voice style
python inference.py --text "Hello, welcome to Supertonic TTS!"

# Specify language and output
python inference.py -t "Hello world" -l en -o output.wav
```

### Command Line Options

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--repo-dir` | `-r` | `supertonic-3` | Repository directory (auto-clones if missing) |
| `--model-dir` | `-m` | `ov_model_fp16` | Directory with OpenVINO models |
| `--config-dir` | `-c` | `<repo-dir>/onnx` | Directory with config files |
| `--voice-style` | `-v` | `<repo-dir>/voice_styles/<first>` | Path to voice style JSON |
| `--text` | `-t` | *required* | Text to synthesize |
| `--lang` | `-l` | `en` | Language code |
| `--output` | `-o` | `output.wav` | Output audio file |
| `--device` | `-d` | `CPU` | OpenVINO device |
| `--precision` | `-p` | `fp32` | Inference precision (GPU only) |
| `--steps` | `-s` | `8` | Diffusion steps |
| `--speed` | | `1.05` | Speech speed multiplier |
| `--benchmark` | `-b` | | Enable benchmark mode |
| `--iterations` | | `10` | Benchmark iterations |

### Examples

```bash
# Simplest usage - auto-selects voice style
python inference.py -t "Hello, world!"

# Specify a particular voice style
python inference.py -t "Hello" -v supertonic-3/voice_styles/speaker_01.json

# GPU inference with FP16
python inference.py -t "Hello" -d GPU -p fp16

# Hindi text
python inference.py -t "नमस्ते दुनिया" -l hi

# Slower speech
python inference.py -t "Slow speech" --speed 0.9

# More diffusion steps (higher quality)
python inference.py -t "High quality" -s 16

# Benchmark performance
python inference.py -t "Benchmark test" -b --iterations 20
```

### Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `hi` | Hindi |
| `ko` | Korean | `ja` | Japanese |
| `de` | German | `fr` | French |
| `es` | Spanish | `it` | Italian |
| `pt` | Portuguese | `ru` | Russian |
| `ar` | Arabic | `zh` | Chinese |

See `helper.py` for the full list of 30+ supported languages.

## Performance Tips

1. **FP16 on GPU**: Use `--precision fp16` for faster GPU inference
2. **FP32 for Accuracy**: Use `--precision fp32` if you notice quality issues
3. **Diffusion Steps**: Lower steps (4-8) are faster, higher (16-32) are better quality
4. **Batch Processing**: Use the Python API for batch inference of multiple texts

## Troubleshooting

### OpenVINO not found
```bash
pip install openvino --upgrade
```

### GPU not detected
```bash
# Check OpenCL installation
clinfo
# Install Intel compute runtime
sudo apt-get install intel-opencl-icd
```

### Model conversion fails
- Ensure ONNX models are in the correct directory
- Check that `ovc` (OpenVINO Model Converter) is in PATH
- Try converting with `--precision fp32` first

## License

See the main project repository for license information.
