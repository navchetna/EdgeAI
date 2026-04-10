"""
Audio processing utilities for OpenVINO Streaming ASR.
Includes VAD, audio conversion, and preprocessing functions.
"""

import io
import struct
import numpy as np
from typing import Tuple, Optional, List
from loguru import logger


class AudioProcessor:
    """
    Audio processing utilities for real-time ASR.
    Handles format conversion, resampling, and preprocessing.
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        
    def decode_wav_bytes(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Decode WAV bytes to numpy array.
        
        Args:
            wav_bytes: Raw WAV file bytes
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            import soundfile as sf
            audio_io = io.BytesIO(wav_bytes)
            audio_array, sample_rate = sf.read(audio_io, dtype="float32")
            return audio_array, sample_rate
        except Exception as e:
            logger.warning(f"soundfile failed, using manual parsing: {e}")
            return self._parse_wav_manual(wav_bytes)
    
    def _parse_wav_manual(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Manually parse WAV bytes without external dependencies.
        Supports 16-bit PCM WAV format.
        """
        if len(wav_bytes) < 44:
            raise ValueError("Invalid WAV: too short")
            
        # Parse RIFF header
        riff = wav_bytes[:4]
        if riff != b'RIFF':
            raise ValueError("Invalid WAV: no RIFF header")
            
        # Parse format chunk
        fmt_chunk = wav_bytes[12:16]
        if fmt_chunk != b'fmt ':
            raise ValueError("Invalid WAV: no fmt chunk")
            
        # Read format info
        audio_format = struct.unpack('<H', wav_bytes[20:22])[0]
        num_channels = struct.unpack('<H', wav_bytes[22:24])[0]
        sample_rate = struct.unpack('<I', wav_bytes[24:28])[0]
        bits_per_sample = struct.unpack('<H', wav_bytes[34:36])[0]
        
        if audio_format != 1:  # PCM
            raise ValueError(f"Unsupported audio format: {audio_format}")
            
        # Find data chunk
        data_start = 44
        for i in range(36, len(wav_bytes) - 8):
            if wav_bytes[i:i+4] == b'data':
                data_start = i + 8
                break
        
        # Read audio data
        raw_audio = wav_bytes[data_start:]
        
        if bits_per_sample == 16:
            audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize to [-1, 1]
        elif bits_per_sample == 32:
            audio = np.frombuffer(raw_audio, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")
            
        # Convert stereo to mono
        if num_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
            
        return audio, sample_rate
    
    def decode_raw_pcm(
        self, 
        pcm_bytes: bytes, 
        sample_rate: int = 16000,
        bits_per_sample: int = 16
    ) -> np.ndarray:
        """
        Decode raw PCM bytes to numpy array.
        
        Args:
            pcm_bytes: Raw PCM audio bytes
            sample_rate: Sample rate of the audio
            bits_per_sample: Bits per sample (8, 16, or 32)
            
        Returns:
            Normalized float32 audio array
        """
        if bits_per_sample == 16:
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0
        elif bits_per_sample == 8:
            audio = np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128) / 128.0
        elif bits_per_sample == 32:
            audio = np.frombuffer(pcm_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")
            
        return audio
    
    def resample(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        Uses linear interpolation for simplicity.
        """
        if orig_sr == target_sr:
            return audio
            
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        
        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        return resampled.astype(np.float32)
    
    def preprocess(
        self, 
        audio_bytes: bytes,
        input_sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        Full preprocessing pipeline for audio bytes.
        
        Args:
            audio_bytes: Raw audio bytes (WAV or PCM)
            input_sample_rate: Sample rate if raw PCM
            
        Returns:
            Preprocessed float32 audio at target sample rate
        """
        # Try decoding as WAV first
        try:
            audio, sample_rate = self.decode_wav_bytes(audio_bytes)
        except Exception:
            # Assume raw PCM
            sample_rate = input_sample_rate or 16000
            audio = self.decode_raw_pcm(audio_bytes, sample_rate)
        
        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to target rate
        if sample_rate != self.target_sample_rate:
            audio = self.resample(audio, sample_rate, self.target_sample_rate)
        
        return audio.astype(np.float32)


class SimpleVAD:
    """
    Simple energy-based Voice Activity Detection.
    For production, consider using Silero VAD or similar.
    """
    
    def __init__(
        self,
        energy_threshold: float = 0.01,
        speech_pad_ms: int = 100,
        min_speech_ms: int = 100,
        sample_rate: int = 16000
    ):
        self.energy_threshold = energy_threshold
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
        
        # State
        self.is_speaking = False
        self.speech_start = 0
        self.silence_start = 0
        
    def process(self, audio: np.ndarray, frame_size_ms: int = 30) -> List[dict]:
        """
        Process audio and detect speech segments.
        
        Args:
            audio: Float32 audio array
            frame_size_ms: Frame size in milliseconds
            
        Returns:
            List of speech segment dictionaries
        """
        frame_size = int(frame_size_ms * self.sample_rate / 1000)
        segments = []
        
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size // 2:
                continue
                
            energy = self._compute_energy(frame)
            is_speech = energy > self.energy_threshold
            
            if is_speech and not self.is_speaking:
                self.is_speaking = True
                self.speech_start = max(0, i - self.speech_pad_samples)
                
            elif not is_speech and self.is_speaking:
                speech_duration = i - self.speech_start
                if speech_duration >= self.min_speech_samples:
                    segments.append({
                        "start": self.speech_start,
                        "end": min(len(audio), i + self.speech_pad_samples),
                        "duration_ms": speech_duration * 1000 / self.sample_rate
                    })
                self.is_speaking = False
        
        # Handle ongoing speech at end
        if self.is_speaking:
            segments.append({
                "start": self.speech_start,
                "end": len(audio),
                "duration_ms": (len(audio) - self.speech_start) * 1000 / self.sample_rate,
                "is_ongoing": True
            })
        
        return segments
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio chunk contains speech."""
        energy = self._compute_energy(audio)
        return energy > self.energy_threshold
    
    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio."""
        return np.sqrt(np.mean(audio ** 2))
    
    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_start = 0
        self.silence_start = 0


class SileroVAD:
    """
    Wrapper for Silero VAD using OpenVINO.
    More accurate than simple energy-based VAD.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "CPU",
        threshold: float = 0.5,
        sample_rate: int = 16000,
        window_size_samples: int = 512
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_size = window_size_samples
        self.device = device
        
        # Try to load Silero VAD model
        self.model = None
        self._load_model(model_path)
        
        # State
        self._h = None
        self._c = None
        
    def _load_model(self, model_path: Optional[str]):
        """Load Silero VAD model."""
        try:
            # Try loading from torch hub first
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self.model = model
            self.get_speech_timestamps = utils[0]
            logger.info("Silero VAD loaded from torch hub")
        except Exception as e:
            logger.warning(f"Could not load Silero VAD: {e}")
            logger.info("Falling back to simple energy-based VAD")
            
    def is_speech(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Check if audio contains speech.
        
        Returns:
            Tuple of (is_speech, confidence)
        """
        if self.model is None:
            # Fallback to energy-based
            energy = np.sqrt(np.mean(audio ** 2))
            return energy > 0.01, float(energy)
        
        try:
            import torch
            audio_tensor = torch.from_numpy(audio).float()
            
            with torch.no_grad():
                probability = self.model(audio_tensor, self.sample_rate).item()
            
            return probability > self.threshold, probability
            
        except Exception as e:
            logger.warning(f"Silero VAD inference failed: {e}")
            energy = np.sqrt(np.mean(audio ** 2))
            return energy > 0.01, float(energy)
    
    def get_speech_segments(
        self, 
        audio: np.ndarray,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100
    ) -> List[dict]:
        """
        Get all speech segments in audio.
        
        Returns:
            List of segment dictionaries with start/end samples
        """
        if self.model is None or self.get_speech_timestamps is None:
            # Fallback
            simple_vad = SimpleVAD(sample_rate=self.sample_rate)
            return simple_vad.process(audio)
        
        try:
            import torch
            audio_tensor = torch.from_numpy(audio).float()
            
            segments = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                return_seconds=False
            )
            
            return [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration_ms": (seg["end"] - seg["start"]) * 1000 / self.sample_rate
                }
                for seg in segments
            ]
            
        except Exception as e:
            logger.warning(f"Silero VAD segmentation failed: {e}")
            simple_vad = SimpleVAD(sample_rate=self.sample_rate)
            return simple_vad.process(audio)


def create_wav_header(
    sample_rate: int,
    num_channels: int = 1,
    bits_per_sample: int = 16,
    audio_length: int = 0
) -> bytes:
    """
    Create WAV file header.
    
    Args:
        sample_rate: Audio sample rate
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample
        audio_length: Length of audio data in bytes
        
    Returns:
        WAV header bytes
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + audio_length,
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1Size
        1,   # AudioFormat (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        audio_length
    )
    
    return header


def audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> bytes:
    """
    Convert numpy audio array to WAV bytes.
    
    Args:
        audio: Float32 audio array normalized to [-1, 1]
        sample_rate: Sample rate
        
    Returns:
        WAV file bytes
    """
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    # Create header
    header = create_wav_header(sample_rate, audio_length=len(audio_bytes))
    
    return header + audio_bytes


def split_audio_chunks(
    audio: np.ndarray,
    chunk_duration_ms: int = 500,
    sample_rate: int = 16000,
    overlap_ms: int = 0
) -> List[np.ndarray]:
    """
    Split audio into chunks for streaming processing.
    
    Args:
        audio: Input audio array
        chunk_duration_ms: Duration of each chunk in ms
        sample_rate: Sample rate
        overlap_ms: Overlap between chunks in ms
        
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration_ms * sample_rate / 1000)
    overlap_samples = int(overlap_ms * sample_rate / 1000)
    step = chunk_samples - overlap_samples
    
    chunks = []
    for i in range(0, len(audio), step):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) >= chunk_samples // 2:  # Include partial chunks
            chunks.append(chunk)
            
    return chunks
