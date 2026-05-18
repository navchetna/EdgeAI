#!/usr/bin/env python3
"""
OpenVINO Inference for Supertonic TTS
Supports CPU/GPU acceleration and FP16/FP32 precision
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov
from openvino import Core 

from helper import (
    AVAILABLE_LANGS,
    Style,
    UnicodeProcessor,
    chunk_text,
    get_latent_mask,
    length_to_mask,
    sanitize_filename,
)

# Default paths
REPO_URL = "https://huggingface.co/Supertone/supertonic-3"
DEFAULT_REPO_DIR = "supertonic-3"
DEFAULT_ONNX_SUBDIR = "onnx"
DEFAULT_VOICE_STYLES_SUBDIR = "voice_styles"


def ensure_repo_exists(repo_dir: str = DEFAULT_REPO_DIR) -> Path:
    """
    Ensure the Supertonic repository exists, clone if missing.

    Args:
        repo_dir: Path to the repository directory

    Returns:
        Path to the repository
    """
    repo_path = Path(repo_dir)

    if not repo_path.exists():
        print(f"Repository not found at '{repo_dir}'. Cloning from Hugging Face...")
        print(f"  URL: {REPO_URL}")
        print()

        try:
            result = subprocess.run(
                ["git", "clone", REPO_URL, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for large repo
            )

            if result.returncode != 0:
                print(f"ERROR: Failed to clone repository")
                print(f"  {result.stderr}")
                raise RuntimeError("Failed to clone Supertonic repository")

            print(f"Repository cloned successfully to '{repo_dir}'")
            print()

        except FileNotFoundError:
            print("ERROR: 'git' command not found. Please install git.")
            raise
        except subprocess.TimeoutExpired:
            print("ERROR: Clone operation timed out.")
            raise

    return repo_path


class TextToSpeechOpenVINO:
    """OpenVINO-based Text-to-Speech inference engine for Supertonic TTS"""

    def __init__(
        self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        core: Core,
        dp_model,
        text_enc_model,
        vector_est_model,
        vocoder_model,
    ):
        """
        Initialize TTS engine with OpenVINO compiled models.

        Args:
            cfgs: Model configuration dictionary
            text_processor: Unicode text processor
            core: OpenVINO Core instance
            dp_model: Compiled duration predictor model
            text_enc_model: Compiled text encoder model
            vector_est_model: Compiled vector estimator model
            vocoder_model: Compiled vocoder model
        """
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.core = core
        self.dp_model = dp_model
        self.text_enc_model = text_enc_model
        self.vector_est_model = vector_est_model
        self.vocoder_model = vocoder_model

        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def sample_noisy_latent(
        self, duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate noisy latent representation based on duration."""
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor
        )
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float = 1.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Internal inference method for batch processing."""
        assert (
            len(text_list) == style.ttl.shape[0]
        ), "Number of texts must match number of style vectors"
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list, lang_list)

        # Duration prediction using OpenVINO
        dp_result = self.dp_model.infer_new_request({
            "text_ids": text_ids,
            "style_dp": style.dp,
            "text_mask": text_mask,
        })
        dur_onnx = list(dp_result.values())[0]
        dur_onnx = dur_onnx / speed

        # Text encoding using OpenVINO
        text_enc_result = self.text_enc_model.infer_new_request({
            "text_ids": text_ids,
            "style_ttl": style.ttl,
            "text_mask": text_mask,
        })
        text_emb_onnx = list(text_enc_result.values())[0]

        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)

        # Vector estimation loop using OpenVINO
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            vector_est_result = self.vector_est_model.infer_new_request({
                "noisy_latent": xt,
                "text_emb": text_emb_onnx,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step_np,
            })
            xt = list(vector_est_result.values())[0]

        # Vocoder using OpenVINO
        vocoder_result = self.vocoder_model.infer_new_request({"latent": xt})
        wav = list(vocoder_result.values())[0]

        return wav, dur_onnx

    def __call__(
        self,
        text: str,
        lang: str,
        style: Style,
        total_step: int,
        speed: float = 1.05,
        silence_duration: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            lang: Language code (e.g., 'en', 'hi', 'ko')
            style: Voice style object
            total_step: Number of diffusion steps
            speed: Speech speed multiplier (default: 1.05)
            silence_duration: Silence between chunks in seconds

        Returns:
            Tuple of (audio waveform, duration)
        """
        assert (
            style.ttl.shape[0] == 1
        ), "Single speaker text to speech only supports single style"
        max_len = 120 if lang in ("ko", "ja") else 300
        text_list = chunk_text(text, max_len=max_len)
        wav_cat = None
        dur_cat = None

        for text_chunk in text_list:
            wav, dur_onnx = self._infer([text_chunk], [lang], style, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur_onnx
            else:
                silence = np.zeros(
                    (1, int(silence_duration * self.sample_rate)), dtype=np.float32
                )
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat += dur_onnx + silence_duration

        return wav_cat, dur_cat

    def batch(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float = 1.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch inference for multiple texts."""
        return self._infer(text_list, lang_list, style, total_step, speed)


def load_openvino_model(core: Core, model_path: Path, device: str, force_fp32: bool = False):
    """
    Load and compile an OpenVINO model.

    Args:
        core: OpenVINO Core instance
        model_path: Path to the .xml model file
        device: Target device (CPU, GPU, etc.)
        force_fp32: Force FP32 execution on GPU

    Returns:
        Compiled model
    """
    model = core.read_model(model_path)

    config = {}
    if "GPU" in device and force_fp32:
        config["INFERENCE_PRECISION_HINT"] = "f32"

    return core.compile_model(model, device, config)


def load_text_to_speech_openvino(
    model_dir: str,
    config_dir: str = None,
    device: str = "CPU",
    force_fp32_on_gpu: bool = True,
) -> TextToSpeechOpenVINO:
    """
    Load Supertonic TTS with OpenVINO backend.

    Args:
        model_dir: Directory containing OpenVINO IR models (.xml/.bin)
        config_dir: Directory containing config files (tts.json, unicode_indexer.json)
                   Defaults to model_dir if not specified
        device: OpenVINO device (CPU, GPU, AUTO, MULTI:CPU,GPU)
        force_fp32_on_gpu: Force FP32 precision on GPU for accuracy

    Returns:
        TextToSpeechOpenVINO instance
    """
    model_dir = Path(model_dir)
    config_dir = Path(config_dir) if config_dir else model_dir

    print(f"Loading Supertonic TTS with OpenVINO on {device}...")

    # Initialize OpenVINO Core
    core = Core()

    # Print available devices
    print(f"  Available devices: {core.available_devices}")

    # GPU precision info
    if "GPU" in device:
        if force_fp32_on_gpu:
            print("  GPU Precision: FP32 (forced for accuracy)")
        else:
            print("  GPU Precision: Default (FP16 if available)")

    # Load configuration
    cfg_path = config_dir / "tts.json"
    with open(cfg_path, "r") as f:
        cfgs = json.load(f)
    print(f"  Loaded config from {cfg_path}")

    # Load text processor
    unicode_indexer_path = config_dir / "unicode_indexer.json"
    text_processor = UnicodeProcessor(str(unicode_indexer_path))
    print(f"  Loaded text processor from {unicode_indexer_path}")

    # Load OpenVINO models
    print("  Loading OpenVINO models...")

    dp_path = model_dir / "duration_predictor.xml"
    text_enc_path = model_dir / "text_encoder.xml"
    vector_est_path = model_dir / "vector_estimator.xml"
    vocoder_path = model_dir / "vocoder.xml"

    dp_model = load_openvino_model(core, dp_path, device, force_fp32_on_gpu)
    print(f"    - Duration predictor loaded")

    text_enc_model = load_openvino_model(core, text_enc_path, device, force_fp32_on_gpu)
    print(f"    - Text encoder loaded")

    vector_est_model = load_openvino_model(core, vector_est_path, device, force_fp32_on_gpu)
    print(f"    - Vector estimator loaded")

    vocoder_model = load_openvino_model(core, vocoder_path, device, force_fp32_on_gpu)
    print(f"    - Vocoder loaded")

    print(f"Model initialized successfully on {device}")

    return TextToSpeechOpenVINO(
        cfgs=cfgs,
        text_processor=text_processor,
        core=core,
        dp_model=dp_model,
        text_enc_model=text_enc_model,
        vector_est_model=vector_est_model,
        vocoder_model=vocoder_model,
    )


def load_voice_style(voice_style_path: str) -> Style:
    """
    Load a voice style from JSON file.

    Args:
        voice_style_path: Path to the voice style JSON file

    Returns:
        Style object
    """
    with open(voice_style_path, "r") as f:
        voice_style = json.load(f)

    ttl_dims = voice_style["style_ttl"]["dims"]
    dp_dims = voice_style["style_dp"]["dims"]

    ttl_data = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).flatten()
    ttl_style = ttl_data.reshape(1, ttl_dims[1], ttl_dims[2])

    dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
    dp_style = dp_data.reshape(1, dp_dims[1], dp_dims[2])

    return Style(ttl_style, dp_style)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Supertonic TTS Inference with OpenVINO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo-dir",
        "-r",
        type=str,
        default=DEFAULT_REPO_DIR,
        help="Supertonic repository directory (will clone if not exists)",
    )

    parser.add_argument(
        "--model-dir",
        "-m",
        type=str,
        default=None,
        help="Directory containing OpenVINO IR models (default: ov_model_fp16)",
    )

    parser.add_argument(
        "--config-dir",
        "-c",
        type=str,
        default=None,
        help=f"Directory containing config files (default: <repo-dir>/{DEFAULT_ONNX_SUBDIR})",
    )

    parser.add_argument(
        "--voice-style",
        "-v",
        type=str,
        default=None,
        help=f"Path to voice style JSON file (default: <repo-dir>/{DEFAULT_VOICE_STYLES_SUBDIR}/<first available>)",
    )

    parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="Text to synthesize",
    )

    parser.add_argument(
        "--lang",
        "-l",
        type=str,
        default="en",
        choices=AVAILABLE_LANGS,
        help="Language code",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="CPU",
        help="OpenVINO device (CPU, GPU, AUTO, MULTI:CPU,GPU)",
    )

    parser.add_argument(
        "--precision",
        "-p",
        type=str,
        choices=["fp16", "fp32"],
        default="fp32",
        help="Inference precision (affects GPU only)",
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=8,
        help="Number of diffusion steps",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.05,
        help="Speech speed multiplier",
    )

    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run benchmark mode (multiple iterations)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )

    return parser.parse_args()


def save_wav(audio: np.ndarray, sample_rate: int, output_path: str):
    """Save audio to WAV file."""
    try:
        import soundfile as sf
        sf.write(output_path, audio.squeeze(), sample_rate)
    except ImportError:
        import scipy.io.wavfile as wavfile
        # Normalize to int16 range
        audio_int16 = (audio.squeeze() * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)


def main():
    args = parse_args()

    # Ensure repository exists
    repo_path = ensure_repo_exists(args.repo_dir)

    # Set default model directory if not specified
    model_dir = args.model_dir
    if model_dir is None:
        model_dir = "ov_model_fp16"
        if not Path(model_dir).exists():
            model_dir = "ov_model_fp32"
        if not Path(model_dir).exists():
            print("ERROR: No OpenVINO model directory found.")
            print("       Please run convert_to_ov.py first to convert ONNX models.")
            print("       Or specify --model-dir explicitly.")
            exit(1)

    # Set default config directory (contains tts.json, unicode_indexer.json)
    config_dir = args.config_dir
    if config_dir is None:
        config_dir = repo_path / DEFAULT_ONNX_SUBDIR

    # Set default voice style if not specified
    voice_style_path = args.voice_style
    if voice_style_path is None:
        voice_styles_dir = repo_path / DEFAULT_VOICE_STYLES_SUBDIR
        if voice_styles_dir.exists():
            # Find first available voice style JSON
            voice_files = list(voice_styles_dir.glob("*.json"))
            if voice_files:
                voice_style_path = str(voice_files[0])
                print(f"Using default voice style: {voice_style_path}")
            else:
                print(f"ERROR: No voice style files found in {voice_styles_dir}")
                exit(1)
        else:
            print(f"ERROR: Voice styles directory not found: {voice_styles_dir}")
            print("       Please specify --voice-style explicitly.")
            exit(1)

    # Determine FP32 forcing based on precision argument
    force_fp32 = args.precision == "fp32"

    # Load TTS model
    tts = load_text_to_speech_openvino(
        model_dir=model_dir,
        config_dir=str(config_dir),
        device=args.device,
        force_fp32_on_gpu=force_fp32,
    )

    # Load voice style
    print(f"Loading voice style from {voice_style_path}...")
    style = load_voice_style(voice_style_path)

    if args.benchmark:
        # Benchmark mode
        print()
        print("=" * 70)
        print("Benchmark Mode")
        print("=" * 70)
        print(f"  Text: {args.text[:50]}...")
        print(f"  Language: {args.lang}")
        print(f"  Iterations: {args.iterations}")
        print()

        # Warmup
        print("Warming up...")
        for _ in range(2):
            _, _ = tts(args.text, args.lang, style, args.steps, args.speed)

        # Benchmark
        print("Running benchmark...")
        times = []
        for i in range(args.iterations):
            start = time.time()
            wav, dur = tts(args.text, args.lang, style, args.steps, args.speed)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.3f}s")

        print()
        print("-" * 70)
        print(f"  Average: {np.mean(times):.3f}s")
        print(f"  Std Dev: {np.std(times):.3f}s")
        print(f"  Min:     {np.min(times):.3f}s")
        print(f"  Max:     {np.max(times):.3f}s")
        print(f"  RTF:     {np.mean(times) / dur[0]:.3f}")
        print("=" * 70)

    else:
        # Single inference mode
        print()
        print(f"Synthesizing: {args.text}")
        print(f"  Language: {args.lang}")
        print(f"  Steps: {args.steps}")
        print(f"  Speed: {args.speed}")
        print()

        start = time.time()
        wav, dur = tts(args.text, args.lang, style, args.steps, args.speed)
        elapsed = time.time() - start

        print(f"  Duration: {dur[0]:.2f}s")
        print(f"  Inference time: {elapsed:.3f}s")
        print(f"  RTF: {elapsed / dur[0]:.3f}")

        # Save output
        save_wav(wav, tts.sample_rate, args.output)
        print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
