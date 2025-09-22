import io 
import time
import soundfile as sf
import openvino_genai as ov_genai


class Whisper:
    def __init__(self, model_name: str, device: str = "GPU"):
        self.pipeline = ov_genai.WhisperPipeline(model_name, device=device)
        self.tokenizer = self.pipeline.get_tokenizer()

    @staticmethod
    def convert_bytes_to_array(audio_bytes: bytes):
        audio_bytes = io.BytesIO(audio_bytes)
        audio_array, sr = sf.read(file=audio_bytes, dtype="float32")
        return audio_array, sr

    def generate(self, audio_array):
        transcription = self.pipeline.generate(audio_array, task="transcribe", return_timestamps=False)
        return transcription