"""
OpenVINO Streaming ASR Server with INT4 Quantization
WebSocket-based real-time speech recognition for edge devices.

Compatible with shrutlekh_v2.html frontend.
"""

import io
import os
import time
import json
import base64
import asyncio
import numpy as np
import soundfile as sf
from loguru import logger
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field

# OpenVINO imports - compatible with multiple versions
try:
    import openvino as ov
    from openvino import Core
    OV_VERSION = ov.__version__ if hasattr(ov, '__version__') else "unknown"
except ImportError:
    # Fallback for older versions
    from openvino.runtime import Core
    import openvino.runtime as ov
    OV_VERSION = getattr(ov, '__version__', "unknown (runtime)")

# Log OpenVINO version
logger.info(f"OpenVINO version: {OV_VERSION}")

# Optional: openvino_tokenizers (not always needed)
try:
    from openvino_tokenizers import convert_tokenizer
    HAS_OV_TOKENIZERS = True
except ImportError:
    HAS_OV_TOKENIZERS = False
    convert_tokenizer = None

# Optional: openvino_genai
HAS_OV_GENAI = False
try:
    import openvino_genai as ov_genai
    HAS_OV_GENAI = True
    logger.info("OpenVINO GenAI available")
except ImportError:
    ov_genai = None
    logger.info("OpenVINO GenAI not available - using fallback")

# WebSocket and server imports
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# SocketIO for compatibility with shrutlekh_v2
import socketio


@dataclass
class StreamingConfig:
    """Configuration for streaming ASR."""
    model_path: str = "whisper_int4"
    device: str = "CPU"  # CPU, GPU, NPU for edge devices
    sample_rate: int = 16000
    chunk_duration_ms: int = 500
    vad_threshold: float = 0.01
    silence_threshold_ms: int = 500
    max_audio_buffer_seconds: float = 30.0
    language: str = "en"
    task: str = "transcribe"


@dataclass
class AudioBuffer:
    """Audio buffer for accumulating streaming audio chunks."""
    samples: deque = field(default_factory=lambda: deque(maxlen=480000))  # 30s @ 16kHz
    last_speech_time: float = 0
    is_speaking: bool = False
    accumulated_audio: list = field(default_factory=list)
    

class StreamingWhisperASR:
    """
    OpenVINO-based Streaming Whisper ASR with INT4 quantization support.
    Designed for edge deployment with low latency.
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.core = ov.Core()
        self.pipeline = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the OpenVINO Whisper model with INT4 quantization."""
        logger.info(f"Loading model from {self.config.model_path} on {self.config.device}")
        
        if HAS_OV_GENAI and os.path.exists(self.config.model_path):
            try:
                # Load the Whisper pipeline with OpenVINO GenAI
                self.pipeline = ov_genai.WhisperPipeline(
                    self.config.model_path,
                    device=self.config.device
                )
                self.tokenizer = self.pipeline.get_tokenizer()
                logger.info("Model loaded successfully with OpenVINO GenAI")
                return
            except Exception as e:
                logger.warning(f"OpenVINO GenAI failed: {e}, using fallback")
        
        # Fallback to custom model loading
        logger.warning("Using custom model implementation")
        self._load_custom_model()
            
    def _load_custom_model(self):
        """Fallback: Load model using core OpenVINO APIs."""
        encoder_path = os.path.join(self.config.model_path, "encoder.xml")
        decoder_path = os.path.join(self.config.model_path, "decoder.xml")
        
        if os.path.exists(encoder_path):
            self.encoder = self.core.compile_model(encoder_path, self.config.device)
            logger.info("Encoder loaded")
            
        if os.path.exists(decoder_path):
            self.decoder = self.core.compile_model(decoder_path, self.config.device)
            logger.info("Decoder loaded")
    
    def preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to normalized float32 array."""
        try:
            audio_io = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(audio_io, dtype="float32")
            
            # Resample if necessary
            if sr != self.config.sample_rate:
                audio_array = self._resample(audio_array, sr, self.config.sample_rate)
            
            # Convert stereo to mono if necessary
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
                
            return audio_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio
            
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def detect_voice_activity(self, audio: np.ndarray) -> bool:
        """Simple energy-based VAD."""
        if len(audio) == 0:
            return False
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > self.config.vad_threshold
    
    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio using OpenVINO Whisper model.
        
        Args:
            audio: Float32 audio array at 16kHz
            language: Optional language code
            
        Returns:
            Dictionary with transcription text and metadata
        """
        if len(audio) == 0:
            return {"text": "", "is_final": False}
            
        start_time = time.perf_counter()
        
        try:
            lang = language or self.config.language
            lang_token = f"<|{lang}|>"
            
            if self.pipeline is not None:
                # Use OpenVINO GenAI pipeline
                result = self.pipeline.generate(
                    audio,
                    task=self.config.task,
                    return_timestamps=False,
                    language=lang_token
                )
                transcription = str(result)
            else:
                # Fallback for custom model
                transcription = self._custom_inference(audio)
                
            latency = time.perf_counter() - start_time
            
            return {
                "text": transcription.strip(),
                "is_final": True,
                "latency": latency,
                "language": lang
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"text": "", "is_final": False, "error": str(e)}
    
    def _custom_inference(self, audio: np.ndarray) -> str:
        """Custom inference path for when openvino_genai is not available."""
        # Placeholder for custom OpenVINO inference logic
        # This would involve manual encoding and decoding steps
        logger.warning("Custom inference not fully implemented")
        return ""

    def transcribe_stream(self, audio_buffer: AudioBuffer) -> Dict[str, Any]:
        """
        Process accumulated audio from streaming buffer.
        
        Args:
            audio_buffer: AudioBuffer containing accumulated samples
            
        Returns:
            Transcription result
        """
        if len(audio_buffer.accumulated_audio) == 0:
            return {"text": "", "is_final": False}
            
        # Concatenate all accumulated audio
        audio = np.concatenate(audio_buffer.accumulated_audio)
        
        # Clear buffer after processing
        audio_buffer.accumulated_audio.clear()
        
        return self.transcribe(audio)


class ASRSessionManager:
    """Manages multiple concurrent ASR sessions."""
    
    def __init__(self, asr_model: StreamingWhisperASR):
        self.asr = asr_model
        self.sessions: Dict[str, AudioBuffer] = {}
        
    def create_session(self, session_id: str) -> AudioBuffer:
        """Create a new ASR session."""
        buffer = AudioBuffer()
        self.sessions[session_id] = buffer
        logger.info(f"Created session: {session_id}")
        return buffer
    
    def get_session(self, session_id: str) -> Optional[AudioBuffer]:
        """Get an existing session."""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed session: {session_id}")
    
    def process_audio_chunk(
        self, 
        session_id: str, 
        audio_bytes: bytes,
        is_final: bool = False
    ) -> Dict[str, Any]:
        """
        Process an audio chunk for a session.
        
        Args:
            session_id: Unique session identifier
            audio_bytes: Raw audio bytes
            is_final: Whether this is the final chunk
            
        Returns:
            Partial or final transcription result
        """
        buffer = self.get_session(session_id)
        if buffer is None:
            buffer = self.create_session(session_id)
        
        # Preprocess and add to buffer
        audio = self.asr.preprocess_audio(audio_bytes)
        
        if len(audio) == 0:
            return {"text": "", "is_final": False}
        
        # Check for voice activity
        has_speech = self.asr.detect_voice_activity(audio)
        current_time = time.time()
        
        if has_speech:
            buffer.is_speaking = True
            buffer.last_speech_time = current_time
            buffer.accumulated_audio.append(audio)
            
            # Return partial result for real-time feedback
            if not is_final:
                return {"text": "", "is_final": False, "has_speech": True}
        
        # Check if we should process (silence detected or final chunk)
        silence_duration = current_time - buffer.last_speech_time
        should_process = (
            is_final or 
            (buffer.is_speaking and 
             silence_duration > self.asr.config.silence_threshold_ms / 1000)
        )
        
        if should_process and len(buffer.accumulated_audio) > 0:
            buffer.is_speaking = False
            result = self.asr.transcribe_stream(buffer)
            result["is_final"] = True
            return result
        
        return {"text": "", "is_final": False}


# Initialize FastAPI app
app = FastAPI(
    title="OpenVINO Streaming ASR",
    description="Real-time speech recognition with INT4 quantization for edge devices",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO for shrutlekh_v2 compatibility
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)

socket_app = socketio.ASGIApp(sio, app)

# Global ASR components (initialized on startup)
asr_model: Optional[StreamingWhisperASR] = None
session_manager: Optional[ASRSessionManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize ASR model on server startup."""
    global asr_model, session_manager
    
    config = StreamingConfig(
        model_path=os.getenv("MODEL_PATH", "whisper_int4"),
        device=os.getenv("DEVICE", "CPU"),
        language=os.getenv("DEFAULT_LANGUAGE", "en")
    )
    
    logger.info("Initializing OpenVINO Streaming ASR...")
    asr_model = StreamingWhisperASR(config)
    session_manager = ASRSessionManager(asr_model)
    logger.info("ASR system ready!")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": asr_model is not None,
        "device": asr_model.config.device if asr_model else None
    }


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription.
    
    Protocol:
    - Client sends JSON: {"audio": "<base64_audio>", "language": "en", "is_final": false}
    - Server responds: {"text": "...", "is_final": true/false, "latency": 0.1}
    """
    await websocket.accept()
    session_id = str(id(websocket))
    
    logger.info(f"WebSocket connection established: {session_id}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "config":
                # Update session configuration
                language = message.get("language", "en")
                logger.info(f"Session {session_id} configured: language={language}")
                await websocket.send_json({"type": "config_ack", "status": "ok"})
                continue
            
            # Process audio
            audio_b64 = message.get("audio", "")
            is_final = message.get("is_final", False)
            language = message.get("language")
            
            if audio_b64:
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_b64)
                
                # Process through ASR
                result = session_manager.process_audio_chunk(
                    session_id, 
                    audio_bytes, 
                    is_final
                )
                
                # Update language if provided
                if language and asr_model:
                    asr_model.config.language = language
                
                # Send result back
                await websocket.send_json(result)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        session_manager.remove_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        session_manager.remove_session(session_id)


# Socket.IO event handlers for shrutlekh_v2 compatibility
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    logger.info(f"Socket.IO client connected: {sid}")
    session_manager.create_session(sid)
    await sio.emit("connect_ack", {"status": "connected"}, to=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"Socket.IO client disconnected: {sid}")
    session_manager.remove_session(sid)


@sio.event
async def start_pipeline(sid, data):
    """
    Handle pipeline start request from shrutlekh_v2.
    
    Expected data format from client:
    {
        "language": "en",
        "task": "transcribe"
    }
    """
    logger.info(f"Pipeline start request from {sid}: {data}")
    
    language = data.get("language", "en")
    if asr_model:
        asr_model.config.language = language
    
    await sio.emit("pipeline_ready", {
        "status": "ready",
        "language": language,
        "model": "whisper_int4"
    }, to=sid)


@sio.event
async def audio_input(sid, data):
    """
    Handle audio input from shrutlekh_v2.
    
    Expected data format:
    {
        "audio": "<base64_encoded_wav>",
        "is_final": false
    }
    
    The client sends audio chunks encoded as base64 WAV.
    """
    try:
        audio_b64 = data.get("audio", "")
        is_final = data.get("is_final", False)
        
        if not audio_b64:
            return
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_b64)
        
        # Process through ASR
        result = session_manager.process_audio_chunk(sid, audio_bytes, is_final)
        
        text = result.get("text", "")
        is_result_final = result.get("is_final", False)
        
        if text or is_result_final:
            # Emit transcription result in shrutlekh_v2 format
            await sio.emit("transcript", {
                "text": text,
                "is_final": is_result_final,
                "latency": result.get("latency", 0)
            }, to=sid)
            
    except Exception as e:
        logger.error(f"Error processing audio from {sid}: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def stop_pipeline(sid, data):
    """Handle pipeline stop request."""
    logger.info(f"Pipeline stop request from {sid}")
    
    # Process any remaining audio in buffer
    buffer = session_manager.get_session(sid)
    if buffer and len(buffer.accumulated_audio) > 0:
        result = asr_model.transcribe_stream(buffer)
        if result.get("text"):
            await sio.emit("transcript", {
                "text": result["text"],
                "is_final": True,
                "latency": result.get("latency", 0)
            }, to=sid)
    
    session_manager.remove_session(sid)
    await sio.emit("pipeline_stopped", {"status": "stopped"}, to=sid)


@sio.event
async def language_change(sid, data):
    """Handle language change request."""
    new_language = data.get("language", "en")
    logger.info(f"Language change request from {sid}: {new_language}")
    
    if asr_model:
        asr_model.config.language = new_language
    
    await sio.emit("language_changed", {
        "language": new_language,
        "status": "ok"
    }, to=sid)


def run_server(host: str = "0.0.0.0", port: int = 8765):
    """Run the streaming ASR server."""
    logger.info(f"Starting OpenVINO Streaming ASR Server on {host}:{port}")
    uvicorn.run(socket_app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenVINO Streaming ASR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--model-path", default="whisper_int4", help="Path to INT4 model")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU"],
                        help="OpenVINO device for inference")
    
    args = parser.parse_args()
    
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["DEVICE"] = args.device
    
    run_server(args.host, args.port)
