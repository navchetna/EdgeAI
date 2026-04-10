"""
OpenVINO Streaming ASR with INT4 Quantization.

A real-time speech recognition system optimized for edge devices,
compatible with shrutlekh_v2 frontend.
"""

from .streaming_server import (
    StreamingConfig,
    StreamingWhisperASR,
    ASRSessionManager,
    run_server
)

from .models import (
    TranscriptionConfig,
    TranscriptionRequest,
    TranscriptionResponse,
    PipelineConfig,
    DeviceType,
    TaskType,
    QuantizationMode
)

from .audio_utils import (
    AudioProcessor,
    SimpleVAD,
    SileroVAD
)

__version__ = "1.0.0"
__all__ = [
    "StreamingConfig",
    "StreamingWhisperASR", 
    "ASRSessionManager",
    "run_server",
    "TranscriptionConfig",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "PipelineConfig",
    "DeviceType",
    "TaskType", 
    "QuantizationMode",
    "AudioProcessor",
    "SimpleVAD",
    "SileroVAD"
]
