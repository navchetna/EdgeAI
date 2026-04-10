#!/usr/bin/env python3
"""
Native preprocessing implementation using librosa
Optimized for Intel platforms (no PyTorch dependency)
"""

import numpy as np
import librosa
import soundfile as sf

class NativeAudioPreprocessor:
    """
    Native audio preprocessor using librosa
    Replaces TorchScript preprocessor for better Intel CPU optimization
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        f_min=0.0,
        f_max=8000.0,
        normalize=True
    ):
        """
        Initialize native preprocessor

        Args:
            sample_rate: Target sample rate (Hz)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            win_length: Window length
            n_mels: Number of mel filters
            f_min: Minimum frequency
            f_max: Maximum frequency
            normalize: Apply mean/std normalization
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize

        # Pre-compute mel filterbank for efficiency
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max
        )

    def load_audio(self, audio_path):
        """
        Load audio file and convert to 16kHz mono

        Args:
            audio_path: Path to audio file

        Returns:
            audio: numpy array (samples,)
        """
        # Load audio
        audio, sr = sf.read(audio_path, dtype='float32')

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                res_type='kaiser_fast'  # Faster resampling
            )

        return audio

    def extract_features(self, audio):
        """
        Extract mel-spectrogram features

        Args:
            audio: numpy array (samples,)

        Returns:
            features: numpy array (n_mels, time)
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True
        )

        # Get magnitude
        magnitude = np.abs(stft)

        # Apply mel filterbank
        mel_spec = np.dot(self.mel_basis, magnitude)

        # Convert to log scale
        log_mel_spec = np.log(mel_spec + 1e-10)

        # Normalize
        if self.normalize:
            mean = np.mean(log_mel_spec, axis=1, keepdims=True)
            std = np.std(log_mel_spec, axis=1, keepdims=True)
            log_mel_spec = (log_mel_spec - mean) / (std + 1e-10)

        return log_mel_spec

    def __call__(self, audio_path):
        """
        Full preprocessing pipeline

        Args:
            audio_path: Path to audio file or numpy array

        Returns:
            features: numpy array (1, n_mels, time)
            length: audio length in samples
        """
        # Load audio if path provided
        if isinstance(audio_path, str):
            audio = self.load_audio(audio_path)
        else:
            audio = audio_path

        # Extract features
        features = self.extract_features(audio)

        # Add batch dimension: (n_mels, time) -> (1, n_mels, time)
        features = features[np.newaxis, :, :]

        return features, len(audio)

    def preprocess_tensor(self, audio_tensor):
        """
        Process audio tensor (for compatibility with torch tensors)

        Args:
            audio_tensor: torch.Tensor or numpy array (1, samples)

        Returns:
            features: numpy array (1, n_mels, time)
            length: numpy array [samples]
        """
        # Convert to numpy if needed
        if hasattr(audio_tensor, 'numpy'):
            audio = audio_tensor.numpy()
        else:
            audio = audio_tensor

        # Remove batch dimension if present
        if audio.ndim > 1:
            audio = audio.squeeze(0)

        # Extract features
        features = self.extract_features(audio)

        # Add batch dimension
        features = features[np.newaxis, :, :]

        return features, np.array([len(audio)])


class OptimizedAudioPreprocessor(NativeAudioPreprocessor):
    """
    Optimized version with caching and batch processing
    """

    def __init__(self, *args, use_cache=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = use_cache
        self._cache = {}

    def __call__(self, audio_path):
        """Process with optional caching"""

        # Check cache
        if self.use_cache and isinstance(audio_path, str):
            if audio_path in self._cache:
                return self._cache[audio_path]

        # Process
        result = super().__call__(audio_path)

        # Cache result
        if self.use_cache and isinstance(audio_path, str):
            self._cache[audio_path] = result

        return result

    def clear_cache(self):
        """Clear feature cache"""
        self._cache.clear()


def benchmark_preprocessing():
    """Benchmark native vs TorchScript preprocessing"""

    import time
    import torch

    print("=" * 70)
    print("Preprocessing Benchmark: Native vs TorchScript")
    print("=" * 70)
    print()

    # Initialize preprocessors
    print("Initializing preprocessors...")
    native_prep = NativeAudioPreprocessor()

    try:
        torch_prep = torch.jit.load('assets/preprocessor.ts', map_location='cpu')
        torch_available = True
    except:
        torch_available = False
        print("Warning: TorchScript preprocessor not found")

    print()

    # Test with different durations
    durations = [1, 5, 10]
    sample_rate = 16000

    print(f"{'Duration':>10} {'Native (ms)':>15} {'TorchScript (ms)':>18} {'Speedup':>12}")
    print("-" * 70)

    for duration in durations:
        num_samples = duration * sample_rate
        audio = np.random.randn(num_samples).astype(np.float32)

        # Benchmark native
        start = time.time()
        for _ in range(10):
            native_prep(audio)
        native_time = (time.time() - start) / 10 * 1000

        # Benchmark TorchScript
        if torch_available:
            audio_torch = torch.from_numpy(audio).unsqueeze(0)
            length_torch = torch.tensor([num_samples])

            start = time.time()
            for _ in range(10):
                with torch.no_grad():
                    torch_prep(input_signal=audio_torch, length=length_torch)
            torch_time = (time.time() - start) / 10 * 1000

            speedup = torch_time / native_time
            print(f"{duration:>10}s {native_time:>15.2f} {torch_time:>18.2f} {speedup:>12.2f}x")
        else:
            print(f"{duration:>10}s {native_time:>15.2f} {'N/A':>18} {'N/A':>12}")

    print()
    print("Note: Speedup > 1.0 means native is faster")
    print()


if __name__ == '__main__':
    print()
    print("Native Audio Preprocessor")
    print("=" * 70)
    print()
    print("This module provides a native Python preprocessing implementation")
    print("that replaces TorchScript for better Intel CPU optimization.")
    print()
    print("Features:")
    print("  - No PyTorch dependency for preprocessing")
    print("  - Uses librosa (optimized with NumPy/SciPy)")
    print("  - Compatible with existing OpenVINO pipeline")
    print("  - Optional caching for repeated files")
    print()
    print("Usage:")
    print("  from native_preprocessing import NativeAudioPreprocessor")
    print("  preprocessor = NativeAudioPreprocessor()")
    print("  features, length = preprocessor('audio.wav')")
    print()

    # Run benchmark
    benchmark_preprocessing()
