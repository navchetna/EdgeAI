"""
Client example and integration helper for OpenVINO Streaming ASR.
Shows how to integrate with shrutlekh_v2 or use standalone.
"""

import asyncio
import base64
import json
import argparse
from pathlib import Path
from typing import Optional

import socketio
import websockets
import numpy as np
from loguru import logger


class StreamingASRClient:
    """
    Client for connecting to OpenVINO Streaming ASR server.
    Can be used standalone or as a library for integration.
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8765",
        language: str = "en",
        use_socketio: bool = True
    ):
        self.server_url = server_url
        self.language = language
        self.use_socketio = use_socketio
        
        self.sio = None
        self.ws = None
        self.is_connected = False
        self.transcription_callback = None
        
    async def connect(self):
        """Connect to the ASR server."""
        if self.use_socketio:
            await self._connect_socketio()
        else:
            await self._connect_websocket()
            
    async def _connect_socketio(self):
        """Connect using Socket.IO (shrutlekh_v2 compatible)."""
        self.sio = socketio.AsyncClient()
        
        @self.sio.event
        async def connect():
            logger.info("Connected to ASR server")
            self.is_connected = True
            
        @self.sio.event
        async def disconnect():
            logger.info("Disconnected from ASR server")
            self.is_connected = False
            
        @self.sio.event
        async def pipeline_ready(data):
            logger.info(f"Pipeline ready: {data}")
            
        @self.sio.event
        async def transcript(data):
            logger.info(f"Transcript: {data}")
            if self.transcription_callback:
                await self.transcription_callback(data)
                
        @self.sio.event
        async def error(data):
            logger.error(f"Server error: {data}")
        
        await self.sio.connect(self.server_url)
        
    async def _connect_websocket(self):
        """Connect using raw WebSocket."""
        ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/transcribe"
        
        self.ws = await websockets.connect(ws_url)
        self.is_connected = True
        logger.info(f"Connected to {ws_url}")
        
    async def start_pipeline(self, language: Optional[str] = None):
        """Start the transcription pipeline."""
        lang = language or self.language
        
        if self.use_socketio and self.sio:
            await self.sio.emit('start_pipeline', {
                'language': lang,
                'task': 'transcribe'
            })
        elif self.ws:
            await self.ws.send(json.dumps({
                'type': 'config',
                'language': lang
            }))
            
        logger.info(f"Pipeline started with language: {lang}")
        
    async def send_audio(
        self,
        audio_bytes: bytes,
        is_final: bool = False
    ):
        """
        Send audio chunk to server.
        
        Args:
            audio_bytes: WAV or raw PCM audio bytes
            is_final: Whether this is the final chunk
        """
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        if self.use_socketio and self.sio:
            await self.sio.emit('audio_input', {
                'audio': audio_b64,
                'is_final': is_final
            })
        elif self.ws:
            await self.ws.send(json.dumps({
                'audio': audio_b64,
                'is_final': is_final
            }))
            
    async def stop_pipeline(self):
        """Stop the transcription pipeline."""
        if self.use_socketio and self.sio:
            await self.sio.emit('stop_pipeline', {})
        
        logger.info("Pipeline stopped")
        
    async def disconnect(self):
        """Disconnect from server."""
        if self.sio:
            await self.sio.disconnect()
        if self.ws:
            await self.ws.close()
            
        self.is_connected = False
        
    def set_transcription_callback(self, callback):
        """Set callback function for transcription results."""
        self.transcription_callback = callback
        
    async def receive_messages(self):
        """Receive messages (for WebSocket mode)."""
        if not self.ws:
            return
            
        try:
            async for message in self.ws:
                data = json.loads(message)
                logger.info(f"Received: {data}")
                if self.transcription_callback:
                    await self.transcription_callback(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")


async def transcribe_file(
    file_path: str,
    server_url: str = "http://localhost:8765",
    language: str = "en",
    chunk_duration_ms: int = 500
):
    """
    Transcribe an audio file using the streaming server.
    
    Args:
        file_path: Path to audio file
        server_url: ASR server URL
        language: Source language
        chunk_duration_ms: Chunk duration for streaming
    """
    import soundfile as sf
    from audio_utils import AudioProcessor, audio_to_wav_bytes, split_audio_chunks
    
    # Load and preprocess audio
    processor = AudioProcessor()
    audio, sr = sf.read(file_path, dtype='float32')
    
    if sr != 16000:
        audio = processor.resample(audio, sr, 16000)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Split into chunks
    chunks = split_audio_chunks(audio, chunk_duration_ms)
    
    # Results storage
    results = []
    
    async def on_transcript(data):
        if data.get('text'):
            results.append(data)
            print(f"[{'FINAL' if data.get('is_final') else 'PARTIAL'}] {data['text']}")
    
    # Connect and transcribe
    client = StreamingASRClient(server_url, language)
    client.set_transcription_callback(on_transcript)
    
    await client.connect()
    await client.start_pipeline(language)
    
    # Send chunks
    for i, chunk in enumerate(chunks):
        is_final = (i == len(chunks) - 1)
        wav_bytes = audio_to_wav_bytes(chunk)
        await client.send_audio(wav_bytes, is_final)
        await asyncio.sleep(chunk_duration_ms / 1000)  # Simulate real-time
    
    # Wait for final results
    await asyncio.sleep(1.0)
    await client.stop_pipeline()
    await client.disconnect()
    
    # Combine final results
    final_text = " ".join(r['text'] for r in results if r.get('is_final'))
    return final_text


async def realtime_microphone(
    server_url: str = "http://localhost:8765",
    language: str = "en"
):
    """
    Real-time transcription from microphone.
    Requires PyAudio or sounddevice.
    """
    try:
        import sounddevice as sd
    except ImportError:
        logger.error("sounddevice not installed. Run: pip install sounddevice")
        return
    
    from audio_utils import audio_to_wav_bytes
    
    sample_rate = 16000
    chunk_duration = 0.5  # 500ms chunks
    chunk_samples = int(sample_rate * chunk_duration)
    
    async def on_transcript(data):
        text = data.get('text', '')
        if text:
            is_final = data.get('is_final', False)
            marker = '✓' if is_final else '...'
            print(f"\r{marker} {text}", end='' if not is_final else '\n')
    
    client = StreamingASRClient(server_url, language)
    client.set_transcription_callback(on_transcript)
    
    await client.connect()
    await client.start_pipeline(language)
    
    print("🎤 Listening... (Press Ctrl+C to stop)")
    
    audio_queue = asyncio.Queue()
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        audio_queue.put_nowait(indata.copy())
    
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocksize=chunk_samples,
            callback=audio_callback
        ):
            while True:
                audio_chunk = await audio_queue.get()
                audio_chunk = audio_chunk.flatten()
                wav_bytes = audio_to_wav_bytes(audio_chunk, sample_rate)
                await client.send_audio(wav_bytes, is_final=False)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
        
    finally:
        await client.stop_pipeline()
        await client.disconnect()


def generate_shrutlekh_patch():
    """
    Generate JavaScript patch for shrutlekh_v2.html to use local server.
    """
    patch = """
// ==========================================
// Patch for shrutlekh_v2.html - Local ASR
// ==========================================
// Add this code in the TranscriptionApp constructor
// to use the local OpenVINO ASR server instead of Bhashini

// Option 1: Replace Bhashini URLs with local server
// Change these lines:
//   this.socketUrl = 'wss://dhruva-api.bhashini.gov.in';
// To:
//   this.socketUrl = 'http://localhost:8765';

// Option 2: Add a toggle for local/remote mode
class LocalASRAdapter {
    constructor(originalApp) {
        this.app = originalApp;
        this.localSocket = null;
        this.useLocal = true; // Toggle this for local/remote
        this.localServerUrl = 'http://localhost:8765';
    }
    
    async connect() {
        if (!this.useLocal) return;
        
        this.localSocket = io(this.localServerUrl, {
            transports: ['websocket', 'polling']
        });
        
        this.localSocket.on('connect', () => {
            console.log('Connected to local ASR server');
        });
        
        this.localSocket.on('pipeline_ready', (data) => {
            console.log('Local pipeline ready:', data);
            // Update status in original app
            this.app.updateStatus('Local ASR Ready', 'ready');
        });
        
        this.localSocket.on('transcript', (data) => {
            // Forward to original app's transcript handler
            this.app.updateTranscript(data.text, data.is_final);
        });
        
        this.localSocket.on('error', (err) => {
            console.error('Local ASR error:', err);
            // Fallback to remote
            this.useLocal = false;
            this.app.showError('Local ASR failed, using remote');
        });
    }
    
    startPipeline(language) {
        if (this.useLocal && this.localSocket) {
            this.localSocket.emit('start_pipeline', {
                language: language,
                task: 'transcribe'
            });
            return true;
        }
        return false;
    }
    
    sendAudio(audioBase64, isFinal) {
        if (this.useLocal && this.localSocket) {
            this.localSocket.emit('audio_input', {
                audio: audioBase64,
                is_final: isFinal
            });
            return true;
        }
        return false;
    }
    
    stopPipeline() {
        if (this.useLocal && this.localSocket) {
            this.localSocket.emit('stop_pipeline', {});
        }
    }
}

// Usage in TranscriptionApp:
// this.localAdapter = new LocalASRAdapter(this);
// await this.localAdapter.connect();
// Then in your audio sending code:
// if (!this.localAdapter.sendAudio(audioB64, isFinal)) {
//     // Fallback to original Bhashini socket
//     this.socket.emit('audio_input', {...});
// }

console.log('Local ASR adapter loaded');
"""
    return patch


async def main():
    parser = argparse.ArgumentParser(description="OpenVINO Streaming ASR Client")
    parser.add_argument("--server", default="http://localhost:8765", help="Server URL")
    parser.add_argument("--language", default="en", help="Language code")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # File transcription
    file_parser = subparsers.add_parser("transcribe", help="Transcribe audio file")
    file_parser.add_argument("file", help="Audio file path")
    
    # Microphone
    mic_parser = subparsers.add_parser("mic", help="Real-time microphone transcription")
    
    # Generate patch
    patch_parser = subparsers.add_parser("patch", help="Generate shrutlekh_v2 patch")
    
    args = parser.parse_args()
    
    if args.command == "transcribe":
        result = await transcribe_file(
            args.file,
            args.server,
            args.language
        )
        print(f"\n=== Final Transcription ===\n{result}")
        
    elif args.command == "mic":
        await realtime_microphone(args.server, args.language)
        
    elif args.command == "patch":
        patch = generate_shrutlekh_patch()
        print(patch)
        
        # Save to file
        with open("shrutlekh_local_asr_patch.js", "w") as f:
            f.write(patch)
        print("\nPatch saved to: shrutlekh_local_asr_patch.js")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
