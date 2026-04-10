"""
Models and data structures for OpenVINO Streaming ASR.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class DeviceType(str, Enum):
    """Supported OpenVINO devices for edge deployment."""
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    AUTO = "AUTO"


class TaskType(str, Enum):
    """Whisper task types."""
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class QuantizationMode(str, Enum):
    """Supported quantization modes."""
    INT4_SYM = "int4_sym"
    INT4_ASYM = "int4_asym"
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


# Pydantic models for API

class TranscriptionConfig(BaseModel):
    """Configuration for transcription requests."""
    language: str = Field(default="en", description="Source language code")
    task: TaskType = Field(default=TaskType.TRANSCRIBE)
    return_timestamps: bool = Field(default=False)
    vad_enabled: bool = Field(default=True, description="Enable Voice Activity Detection")


class TranscriptionRequest(BaseModel):
    """Request model for transcription API."""
    audio: str = Field(..., description="Base64 encoded audio data")
    config: Optional[TranscriptionConfig] = None
    session_id: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription results."""
    text: str
    is_final: bool = False
    latency: Optional[float] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    timestamps: Optional[List[Dict[str, Any]]] = None


class StreamingMessage(BaseModel):
    """WebSocket message format."""
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PipelineConfig(BaseModel):
    """Configuration for ASR pipeline."""
    model_path: str = "whisper_int4"
    device: DeviceType = DeviceType.CPU
    sample_rate: int = 16000
    language: str = "en"
    task: TaskType = TaskType.TRANSCRIBE
    
    # Streaming parameters
    chunk_duration_ms: int = 500
    silence_threshold_ms: int = 500
    vad_threshold: float = 0.01
    max_buffer_seconds: float = 30.0
    
    # Model parameters
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0


class SessionInfo(BaseModel):
    """Information about an active ASR session."""
    session_id: str
    created_at: datetime
    language: str
    device: str
    is_active: bool = True
    total_audio_seconds: float = 0.0
    total_transcriptions: int = 0


class HealthStatus(BaseModel):
    """Server health status."""
    status: str
    model_loaded: bool
    device: Optional[str] = None
    active_sessions: int = 0
    uptime_seconds: float = 0.0
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response format."""
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None


# Dataclasses for internal use

@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    data: bytes
    sample_rate: int
    timestamp: float
    duration_ms: float
    has_speech: bool = False


@dataclass
class TranscriptionResult:
    """Internal transcription result."""
    text: str
    is_final: bool
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float = 1.0
    language: Optional[str] = None
    latency_ms: float = 0.0

    def to_response(self) -> TranscriptionResponse:
        """Convert to API response format."""
        return TranscriptionResponse(
            text=self.text,
            is_final=self.is_final,
            latency=self.latency_ms / 1000,
            language=self.language,
            confidence=self.confidence
        )


@dataclass 
class VADResult:
    """Voice Activity Detection result."""
    is_speech: bool
    speech_probability: float
    start_sample: int
    end_sample: int


@dataclass
class ModelInfo:
    """Information about loaded model."""
    name: str
    path: str
    device: str
    quantization: QuantizationMode
    size_mb: float
    supported_languages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "device": self.device,
            "quantization": self.quantization.value,
            "size_mb": self.size_mb,
            "supported_languages": self.supported_languages
        }


# Language mappings

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "doi": "Dogri",
    "mni": "Manipuri",
    "sat": "Santali",
    "mai": "Maithili",
    "brx": "Bodo",
    "gom": "Konkani",
    # Common international languages
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
}


def get_language_token(language_code: str) -> str:
    """Get Whisper language token for a language code."""
    return f"<|{language_code}|>"


def validate_language(language_code: str) -> bool:
    """Check if a language code is supported."""
    return language_code.lower() in SUPPORTED_LANGUAGES
