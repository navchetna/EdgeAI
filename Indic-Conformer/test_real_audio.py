"""
Test with Real Audio Files
Transcribe actual audio files in supported Indian languages
Supports CPU, GPU, and AUTO device selection
"""

from openvino_inference_optimized import IndicASROpenVINOOptimized
import time
import sys
import os

SUPPORTED_LANGUAGES = {
    'as': 'Assamese',
    'bn': 'Bengali',
    'brx': 'Bodo',
    'doi': 'Dogri',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'kok': 'Konkani',
    'ks': 'Kashmiri',
    'mai': 'Maithili',
    'ml': 'Malayalam',
    'mni': 'Manipuri',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'or': 'Odia',
    'pa': 'Punjabi',
    'sa': 'Sanskrit',
    'sat': 'Santali',
    'sd': 'Sindhi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ur': 'Urdu'
}

if len(sys.argv) < 3:
    print("Usage: python test_real_audio.py <audio_file> <language_code> [device] [precision]")
    print()
    print("Arguments:")
    print("  audio_file      : Path to audio file (WAV, MP3, FLAC, etc.)")
    print("  language_code   : Language code (see below)")
    print("  device          : CPU, GPU, or AUTO (optional, default: CPU)")
    print("  precision       : fp16 or fp32 (optional, auto-select if not specified)")
    print()
    print("Supported languages:")
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        print(f"  {code:4s} - {name}")
    print()
    print("Device options:")
    print("  CPU         - Use Intel CPU")
    print("  GPU         - Use Intel GPU (if available)")
    print("  AUTO        - Automatically select best device")
    print("  MULTI:CPU,GPU - Use both CPU and GPU")
    print()
    print("Precision (auto-selected by default):")
    print("  fp16        - Faster GPU, smaller models (2.4 GB)")
    print("  fp32        - Better accuracy, larger models (4.8 GB)")
    print("  auto        - CPU uses fp32 if available, GPU uses fp16 (recommended)")
    print()
    print("Examples:")
    print("  python test_real_audio.py speech.wav hi")
    print("  python test_real_audio.py speech.wav hi CPU")
    print("  python test_real_audio.py speech.wav hi GPU")
    print("  python test_real_audio.py speech.wav hi GPU fp16")
    print("  python test_real_audio.py speech.wav hi GPU fp32")
    print("  python test_real_audio.py audio.mp3 bn CPU fp32")
    sys.exit(1)

audio_file = sys.argv[1]
lang = sys.argv[2]
device = sys.argv[3].upper() if len(sys.argv) > 3 else 'CPU'
precision = sys.argv[4].lower() if len(sys.argv) > 4 else 'auto'

# Validate inputs
if not os.path.exists(audio_file):
    print(f"ERROR: Audio file not found: {audio_file}")
    sys.exit(1)

if lang not in SUPPORTED_LANGUAGES:
    print(f"ERROR: Unsupported language code: {lang}")
    print(f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")
    sys.exit(1)

# Validate device
valid_devices = ['CPU', 'GPU', 'AUTO']
if not any(device.startswith(prefix) for prefix in valid_devices + ['MULTI:']):
    print(f"ERROR: Unsupported device: {device}")
    print(f"Supported: CPU, GPU, AUTO, MULTI:CPU,GPU")
    sys.exit(1)

# Validate and auto-select precision
if precision not in ['auto', 'fp16', 'fp32']:
    print(f"ERROR: Invalid precision: {precision}")
    print(f"Supported: auto, fp16, fp32")
    sys.exit(1)

# Check which model directories exist
fp16_available = os.path.exists('openvino_models')
fp32_available = os.path.exists('openvino_models_fp32')

if not fp16_available and not fp32_available:
    print("ERROR: No model directories found!")
    print("Expected: openvino_models/ or openvino_models_fp32/")
    print("Run: python convert_to_openvino.py")
    sys.exit(1)

# Auto-select model directory based on device and precision
if precision == 'auto':
    # Auto-select optimal precision for each device
    if 'GPU' in device:
        # GPU: prefer FP16 for speed
        if fp16_available:
            model_dir = 'openvino_models'
            actual_precision = 'fp16'
            reason = 'FP16 selected for GPU speed'
        elif fp32_available:
            model_dir = 'openvino_models_fp32'
            actual_precision = 'fp32'
            reason = 'FP32 (FP16 not available)'
        else:
            print("ERROR: No models found")
            sys.exit(1)
    else:
        # CPU: prefer FP32 for accuracy
        if fp32_available:
            model_dir = 'openvino_models_fp32'
            actual_precision = 'fp32'
            reason = 'FP32 selected for CPU accuracy'
        elif fp16_available:
            model_dir = 'openvino_models'
            actual_precision = 'fp16'
            reason = 'FP16 (CPU will convert to FP32 automatically)'
        else:
            print("ERROR: No models found")
            sys.exit(1)
elif precision == 'fp16':
    if fp16_available:
        model_dir = 'openvino_models'
        actual_precision = 'fp16'
        reason = 'FP16 explicitly requested'
    else:
        print("ERROR: FP16 models not found (openvino_models/ directory)")
        print("Run: python convert_to_openvino.py")
        sys.exit(1)
elif precision == 'fp32':
    if fp32_available:
        model_dir = 'openvino_models_fp32'
        actual_precision = 'fp32'
        reason = 'FP32 explicitly requested'
    else:
        print("ERROR: FP32 models not found (openvino_models_fp32/ directory)")
        print("Run: python convert_to_openvino_fp32.py")
        sys.exit(1)

print("=" * 80)
print(f"Testing Inference - {SUPPORTED_LANGUAGES[lang]}")
print("=" * 80)
print()

print(f"Configuration:")
print(f"  Audio file: {audio_file}")
print(f"  Language: {SUPPORTED_LANGUAGES[lang]} ({lang})")
print(f"  Device: {device}")
print(f"  Precision: {actual_precision.upper()} ({reason})")
print(f"  Model directory: {model_dir}")
print()

# Display available models
print(f"Available models:")
if fp16_available:
    print(f"  ✓ FP16 models (openvino_models/) - 2.4 GB")
else:
    print(f"  ✗ FP16 models not found")
if fp32_available:
    print(f"  ✓ FP32 models (openvino_models_fp32/) - 4.8 GB")
else:
    print(f"  ✗ FP32 models not found")
print()

print(f"Loading {actual_precision.upper()} model on {device}...")
try:
    model = IndicASROpenVINOOptimized(
        model_dir=model_dir,  # Use selected model directory
        device=device,
        use_native_preprocessing=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Failed to load model on {device}")
    print(f"Details: {e}")
    print()
    if device == 'GPU':
        print("Troubleshooting:")
        print("  1. Check if Intel GPU is available")
        print("  2. Try 'CPU' device instead")
        print("  3. Try 'AUTO' to let OpenVINO choose")
    sys.exit(1)
print()

# CTC decoding
print("-" * 80)
print("CTC Decoding (Greedy):")
print("-" * 80)
start = time.time()
text = model.transcribe(audio_file, lang=lang, decoding='ctc')
elapsed = time.time() - start
print(f"Result: {text}")
print(f"Time: {elapsed:.3f} seconds")
print()

# Calculate RTF if possible
try:
    import librosa
    audio_data, sr = librosa.load(audio_file, sr=16000)
    duration = len(audio_data) / sr
    rtf = elapsed / duration
    print("=" * 80)
    print("Performance:")
    print("=" * 80)
    print(f"Audio duration: {duration:.2f}s")
    print(f"Processing time: {elapsed:.3f}s")
    print(f"RTF: {rtf:.3f}")
    print(f"Speedup: {1/rtf:.1f}x real-time")
except:
    pass

print()
print("=" * 80)
print(f"Summary:")
print(f"  Device: {device}")
print(f"  Precision: {actual_precision.upper()}")
print(f"  Model directory: {model_dir}")
print(f"  Decoding: CTC (RNNT not implemented)")
print()
if actual_precision == 'fp16':
    if 'GPU' in device:
        print(f"  Note: GPU uses FP16 directly (fast, good accuracy)")
    else:
        print(f"  Note: CPU converts FP16→FP32 automatically (accurate)")
else:
    print(f"  Note: Using FP32 for maximum accuracy")
print("=" * 80)
