# OpenVINO Streaming ASR with INT4 Quantization

Real-time speech recognition optimized for edge devices using OpenVINO with INT4 weight compression. Compatible with **shrutlekh_v2** for live transcription and translation.

## Features

- **INT4 Quantization**: 4x model size reduction with minimal accuracy loss
- **Streaming Support**: Real-time transcription via WebSocket/Socket.IO
- **Edge Optimized**: Runs on CPU, GPU, and NPU (Intel AI PCs)
- **Multi-language**: Supports 20+ Indian and international languages
- **VAD Integration**: Voice Activity Detection for efficient processing
- **shrutlekh_v2 Compatible**: Drop-in replacement for Bhashini API

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     shrutlekh_v2.html                       │
│                   (Frontend Browser)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │ WebSocket/Socket.IO
                      │ (Base64 Audio Chunks)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              streaming_server.py                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Socket.IO   │  │   Session   │  │  Audio Processor    │  │
│  │  Events     │◄─┤   Manager   │◄─┤  & VAD              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           StreamingWhisperASR                            ││
│  │  ┌──────────────┐    ┌──────────────────────────────┐   ││
│  │  │ OpenVINO     │    │  Whisper INT4 Model          │   ││
│  │  │ GenAI        │───▶│  (Encoder + Decoder)         │   ││
│  │  │ Pipeline     │    │  CPU / GPU / NPU             │   ││
│  │  └──────────────┘    └──────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd EdgeAI/StreamingASR
pip install -r requirements.txt
```

### 2. Quantize Whisper Model to INT4

```bash
# Using Optimum Intel (recommended)
python quantize_int4.py --model small --use-optimum --output-dir whisper_int4

# Using NNCF with calibration
python quantize_int4.py --model small --config cpu_balanced --num-calibration-samples 50

# For NPU deployment
python quantize_int4.py --model small --config npu_optimized --device npu
```

### 3. Start the Server

```bash
# With default settings (CPU, port 8765)
python streaming_server.py

# With custom configuration
python streaming_server.py --device GPU --port 8765 --model-path whisper_int4
```

### 4. Configure shrutlekh_v2.html

Update the socket URL in `shrutlekh_v2.html` to point to your local server:

```javascript
// Replace the Bhashini API URL with local server
this.socketUrl = 'ws://localhost:8765';
```

## Quantization Options

### Model Sizes

| Model | Parameters | INT4 Size | FP32 Size | RTF (CPU) |
|-------|------------|-----------|-----------|-----------|
| tiny  | 39M        | ~20MB     | ~150MB    | ~0.1      |
| base  | 74M        | ~40MB     | ~290MB    | ~0.15     |
| small | 244M       | ~130MB    | ~970MB    | ~0.3      |
| medium| 769M       | ~400MB    | ~3GB      | ~0.5      |
| large-v3 | 1.5B    | ~800MB    | ~6GB      | ~0.8      |

*RTF = Real-Time Factor (lower is faster)*

### Quantization Configs

```python
# cpu_low_power - Best for battery-powered devices
{
    "mode": "INT4_ASYM",
    "group_size": 128,
    "ratio": 1.0
}

# cpu_balanced - Good accuracy/speed trade-off
{
    "mode": "INT4_SYM", 
    "group_size": 64,
    "ratio": 0.8
}

# npu_optimized - Intel NPU acceleration
{
    "mode": "INT4_SYM",
    "group_size": 128,
    "all_layers": True
}

# gpu_hybrid - For integrated/discrete GPUs
{
    "mode": "INT4_ASYM",
    "group_size": 32,
    "ratio": 0.9
}
```

## API Reference

### Socket.IO Events (shrutlekh_v2 compatible)

**Client → Server:**

```javascript
// Start transcription pipeline
socket.emit('start_pipeline', {
    language: 'en',
    task: 'transcribe'
});

// Send audio chunk
socket.emit('audio_input', {
    audio: '<base64_encoded_wav>',
    is_final: false
});

// Stop pipeline
socket.emit('stop_pipeline', {});

// Change language
socket.emit('language_change', {
    language: 'hi'
});
```

**Server → Client:**

```javascript
// Pipeline ready
socket.on('pipeline_ready', (data) => {
    // { status: 'ready', language: 'en', model: 'whisper_int4' }
});

// Transcription result
socket.on('transcript', (data) => {
    // { text: 'Hello world', is_final: true, latency: 0.15 }
});
```

### WebSocket API

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8765/ws/transcribe');

// Send audio
ws.send(JSON.stringify({
    audio: '<base64_audio>',
    language: 'en',
    is_final: false
}));

// Receive transcription
ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log(result.text); // Transcription
};
```

### REST Endpoints

```bash
# Health check
GET /health

# Response:
{
    "status": "healthy",
    "model_loaded": true,
    "device": "CPU"
}
```

## Integration with shrutlekh_v2

### Option 1: Direct Replacement

Replace the Bhashini WebSocket URL with the local server:

```javascript
// In shrutlekh_v2.html, modify the TranscriptionApp class:
this.socketUrl = 'ws://localhost:8765';
```

### Option 2: Hybrid Mode

Use local ASR with Bhashini for translation:

```javascript
// Keep Bhashini for translation
this.translationUrl = 'wss://dhruva-api.bhashini.gov.in';

// Use local for ASR
this.asrSocket = io('http://localhost:8765');
```

### Option 3: Full Offline Mode

For completely offline operation, implement local translation using OpenVINO-optimized IndicTrans2 models.

## Performance Tuning

### CPU Optimization

```bash
# Enable Intel optimizations
export OV_CPU_NSTREAMS=4
export OV_CPU_THREADS_NUM=8

# Use performance hint
python streaming_server.py --device CPU
```

### GPU Optimization

```bash
# For Intel integrated GPU
python streaming_server.py --device GPU

# Set GPU streams for throughput
export OV_GPU_NSTREAMS=2
```

### NPU Optimization (Intel Core Ultra)

```bash
# Leverage NPU for inference
python streaming_server.py --device NPU

# Ensure INT4 model for NPU compatibility
python quantize_int4.py --model small --config npu_optimized
```

## Troubleshooting

### Model Not Found
```
Error: Model path 'whisper_int4' does not exist
```
Solution: Run the quantization script first:
```bash
python quantize_int4.py --model small --use-optimum
```

### WebSocket Connection Failed
```
WebSocket connection to 'ws://localhost:8765' failed
```
Solution: Check if server is running and port is not blocked:
```bash
netstat -an | findstr 8765
```

### Slow Inference on CPU
Solution: Use smaller model or enable threading:
```bash
export OV_CPU_THREADS_NUM=4
python streaming_server.py --model-path whisper_tiny_int4
```

### CUDA/GPU Not Detected
Solution: Install OpenVINO GPU plugin:
```bash
pip install openvino-dev[gpu]
```

## Project Structure

```
StreamingASR/
├── streaming_server.py    # Main WebSocket server
├── quantize_int4.py       # INT4 quantization script
├── models.py              # Pydantic models & schemas
├── audio_utils.py         # Audio processing & VAD
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── whisper_int4/          # Quantized model (after running quantization)
    ├── encoder/
    ├── decoder/
    ├── tokenizer.json
    └── config.json
```

## License

This project is for research and development purposes. Whisper models are subject to OpenAI's license terms.

## References

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [NNCF Quantization Guide](https://docs.openvino.ai/nncf/)
- [Whisper Model](https://github.com/openai/whisper)
- [Bhashini Platform](https://bhashini.gov.in/)
