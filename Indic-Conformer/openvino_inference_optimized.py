#!/usr/bin/env python3
"""
Optimized OpenVINO Inference with Native Preprocessing and GPU Support
PRODUCTION VERSION - No PyTorch dependency
"""

import numpy as np
import json
from pathlib import Path
from openvino.runtime import Core
from native_preprocessing import NativeAudioPreprocessor

class IndicASROpenVINOOptimized:
    """
    Optimized OpenVINO inference engine with:
    - Native preprocessing (no PyTorch for preprocessing)
    - GPU support
    - Multi-device support
    """

    def __init__(
        self,
        model_dir="openvino_models",
        config_path=".",
        device="CPU",
        use_native_preprocessing=True,
        force_fp32_on_gpu=True
    ):
        """
        Initialize optimized OpenVINO inference engine

        Args:
            model_dir: Directory containing OpenVINO IR models
            config_path: Path to config.json and vocab files
            device: OpenVINO device (CPU, GPU, AUTO, MULTI:CPU,GPU)
            use_native_preprocessing: DEPRECATED - always True (kept for compatibility)
            force_fp32_on_gpu: Force FP32 execution on GPU for accuracy (default: True)
        """
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)
        self.device = device
        self.force_fp32_on_gpu = force_fp32_on_gpu

        # Force native preprocessing (PyTorch-free)
        if not use_native_preprocessing:
            print("WARNING: TorchScript preprocessing no longer supported.")
            print("         Using native preprocessing instead (PyTorch-free).")
        self.use_native_preprocessing = True

        # Initialize OpenVINO
        print(f"Initializing OpenVINO on {device}...")
        self.core = Core()
        self.compiled_models = {}

        # Print available devices
        self._print_available_devices()

        # GPU precision configuration
        if 'GPU' in device:
            if self.force_fp32_on_gpu:
                print("  GPU Precision: FP32 (forced for accuracy)")
            else:
                print("  GPU Precision: FP16 (default, faster but less accurate)")

        # Load configuration
        self._load_config()

        # Load vocabulary and language masks
        self._load_vocab()

        # Load models
        self._load_models()

        # Load native preprocessor (PyTorch-free)
        print("  Using native preprocessing (librosa + NumPy, no PyTorch)")
        self.preprocessor = NativeAudioPreprocessor(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0
        )
        self.preprocessing_type = 'native'

        print(f"Model initialized successfully on {device}")

    def _print_available_devices(self):
        """Print available OpenVINO devices"""
        devices = self.core.available_devices
        print(f"  Available devices: {devices}")

        for device in devices:
            try:
                device_name = self.core.get_property(device, "FULL_DEVICE_NAME")
                print(f"    - {device}: {device_name}")
            except:
                print(f"    - {device}")

    def _load_config(self):
        """Load model configuration"""
        config_file = self.config_path / "config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.BLANK_ID = config.get('BLANK_ID', 256)
        self.SOS = config.get('SOS', 256)
        self.RNNT_MAX_SYMBOLS = config.get('RNNT_MAX_SYMBOLS', 10)
        self.PRED_RNN_LAYERS = config.get('PRED_RNN_LAYERS', 2)
        self.PRED_RNN_HIDDEN_DIM = config.get('PRED_RNN_HIDDEN_DIM', 640)
        self.FRAME_DURATION_MS = config.get('FRAME_DURATION_MS', 0.08)

    def _load_vocab(self):
        """Load vocabulary and language masks"""
        vocab_file = self.config_path / "assets" / "vocab.json"
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        masks_file = self.config_path / "assets" / "language_masks.json"
        with open(masks_file, 'r') as f:
            self.language_masks = json.load(f)

    def _load_models(self):
        """Load all OpenVINO IR models"""
        print("Loading OpenVINO models...")

        # Core models
        core_models = [
            'encoder',
            'ctc_decoder',
            'rnnt_decoder',
            'joint_enc',
            'joint_pred',
            'joint_pre_net'
        ]

        # Track required models for CTC
        required_models = ['encoder', 'ctc_decoder']
        missing_required = []

        for model_name in core_models:
            xml_path = self.model_dir / model_name / f"{model_name}.xml"
            if xml_path.exists():
                try:
                    model = self.core.read_model(xml_path)

                    # Configure device-specific settings for accuracy
                    config = {}
                    if 'GPU' in self.device and self.force_fp32_on_gpu:
                        # Force FP32 execution on GPU for accuracy
                        config['INFERENCE_PRECISION_HINT'] = 'f32'
                        print(f"  GPU: Forcing FP32 execution for {model_name}")

                    # Compile model with config
                    self.compiled_models[model_name] = self.core.compile_model(
                        model, self.device, config
                    )
                    print(f"  Loaded: {model_name} on {self.device}")
                except Exception as e:
                    print(f"  ERROR: Failed to load {model_name}")
                    print(f"         {e}")
                    if model_name in required_models:
                        missing_required.append(model_name)
            else:
                print(f"  WARNING: {model_name} not found at {xml_path}")
                if model_name in required_models:
                    missing_required.append(model_name)

        # Check if required models loaded
        if missing_required:
            raise RuntimeError(
                f"Required models failed to load: {missing_required}\n"
                f"Model directory: {self.model_dir}\n"
                f"Make sure models are converted and directory exists."
            )

        # Language-specific models (load on-demand or pre-load)
        languages = list(self.vocab.keys())
        for lang in languages:
            model_name = f"joint_post_net_{lang}"
            xml_path = self.model_dir / model_name / f"{model_name}.xml"
            if xml_path.exists():
                model = self.core.read_model(xml_path)

                # Use same config as core models for consistency
                config = {}
                if 'GPU' in self.device and self.force_fp32_on_gpu:
                    config['INFERENCE_PRECISION_HINT'] = 'f32'

                self.compiled_models[model_name] = self.core.compile_model(
                    model, self.device, config
                )

        print(f"Total models loaded: {len(self.compiled_models)}")

    def preprocess_audio(self, audio_input):
        """
        Preprocess audio using native librosa preprocessing (PyTorch-free)

        Args:
            audio_input: File path or numpy array

        Returns:
            audio_signal: np.ndarray
            length: np.ndarray
        """
        # Native preprocessing only
        if isinstance(audio_input, str):
            features, length = self.preprocessor(audio_input)
        else:
            features, length = self.preprocessor.preprocess_tensor(audio_input)

        return features, np.array([length])

    def encode(self, audio_features, audio_length):
        """Run encoder inference"""
        encoder = self.compiled_models['encoder']
        infer_request = encoder.create_infer_request()

        input_names = [inp.any_name for inp in encoder.inputs]

        inputs = {
            input_names[0]: audio_features,
            input_names[1]: audio_length
        }

        infer_request.infer(inputs)

        outputs = infer_request.get_output_tensor(0).data
        encoded_lengths = infer_request.get_output_tensor(1).data

        return outputs, encoded_lengths

    def ctc_decode(self, encoder_outputs, encoded_lengths, lang):
        """CTC decoding"""
        ctc_decoder = self.compiled_models['ctc_decoder']
        infer_request = ctc_decoder.create_infer_request()

        input_name = ctc_decoder.inputs[0].any_name
        infer_request.infer({input_name: encoder_outputs})

        logprobs = infer_request.get_output_tensor(0).data

        # Apply language mask
        mask = np.array(self.language_masks[lang])
        masked_logprobs = logprobs[:, :, mask]

        # Apply log-softmax
        exp_logprobs = np.exp(masked_logprobs)
        sum_exp = np.sum(exp_logprobs, axis=-1, keepdims=True)
        log_softmax = masked_logprobs - np.log(sum_exp)

        # Greedy decoding
        indices = np.argmax(log_softmax[0], axis=-1)

        # Collapse consecutive duplicates
        collapsed = []
        prev = None
        for idx in indices:
            if idx != prev:
                collapsed.append(idx)
                prev = idx

        # Convert to text
        text_tokens = []
        for idx in collapsed:
            if idx != self.BLANK_ID:
                text_tokens.append(self.vocab[lang][idx])

        # Join and clean
        text = ''.join(text_tokens).replace('\u2581', ' ').strip()
        return text

    def transcribe(self, audio_input, lang, decoding='ctc'):
        """
        Full transcription pipeline (PyTorch-free)

        Args:
            audio_input: File path or numpy array
            lang: Language code
            decoding: 'ctc' (rnnt not implemented)

        Returns:
            transcription: Text string
        """
        # Preprocess
        audio_features, audio_length = self.preprocess_audio(audio_input)

        # Encode
        encoder_outputs, encoded_lengths = self.encode(
            audio_features, audio_length
        )

        # Decode
        if decoding == 'ctc':
            return self.ctc_decode(encoder_outputs, encoded_lengths, lang)
        elif decoding == 'rnnt':
            raise NotImplementedError("RNNT decoding not yet implemented")
        else:
            raise ValueError(f"Unknown decoding method: {decoding}")

    def get_supported_languages(self):
        """Get list of supported languages"""
        return list(self.vocab.keys())

    def get_model_info(self):
        """Get model information"""
        return {
            'device': self.device,
            'preprocessing': self.preprocessing_type,
            'blank_id': self.BLANK_ID,
            'sos': self.SOS,
            'languages': self.get_supported_languages(),
            'models_loaded': len(self.compiled_models),
            'available_devices': self.core.available_devices
        }


if __name__ == '__main__':
    print("=" * 70)
    print("Optimized OpenVINO Indic Conformer ASR")
    print("=" * 70)
    print()

    # Test with native preprocessing
    print("Testing with NATIVE preprocessing (recommended):")
    print("-" * 70)
    model = IndicASROpenVINOOptimized(device="CPU", use_native_preprocessing=True)

    info = model.get_model_info()
    print()
    print("Model Information:")
    print(f"  Device: {info['device']}")
    print(f"  Preprocessing: {info['preprocessing']}")
    print(f"  Available devices: {info['available_devices']}")
    print(f"  Languages: {len(info['languages'])}")
    print(f"  Models loaded: {info['models_loaded']}")
    print()

    print("=" * 70)
    print()
    print("Usage Examples:")
    print()
    print("1. CPU with native preprocessing (RECOMMENDED):")
    print("   model = IndicASROpenVINOOptimized(device='CPU', use_native_preprocessing=True)")
    print()
    print("2. GPU (if available):")
    print("   model = IndicASROpenVINOOptimized(device='GPU', use_native_preprocessing=True)")
    print()
    print("3. AUTO device selection:")
    print("   model = IndicASROpenVINOOptimized(device='AUTO', use_native_preprocessing=True)")
    print()
    print("4. Multi-device:")
    print("   model = IndicASROpenVINOOptimized(device='MULTI:CPU,GPU', use_native_preprocessing=True)")
    print()
    print("Transcribe:")
    print("   text = model.transcribe('audio.wav', lang='hi', decoding='ctc')")
    print()
