import io
import numpy as np
import soundfile as sf
import torch
import json
import time
from loguru import logger
import openvino as ov
from kokoro import KPipeline
from kokoro.model import KModel
from pathlib import Path

core = ov.Core()
MAX_BUFFER_LENGTH = 100


class OVKModel(KModel):
    def __init__(self, model_dir: Path, device: str):
        torch.nn.Module.__init__(self)
        with (Path(model_dir) / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        self.vocab = config["vocab"]
        self.model = core.compile_model(Path(model_dir) / "openvino_model.xml", "CPU")
        self.context_length = config["plbert"]["max_position_embeddings"]


    @property
    def device(self):
        return torch.device("cpu")
    
    def forward_with_tokens(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1) -> tuple[torch.FloatTensor, torch.LongTensor]:
        outputs = self.model([input_ids, ref_s, torch.tensor(speed)])
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])
    

class TTS:
    def __init__(self, model_dir: str, device: str, model_id: str = "hexgrad/Kokoro-82M"):
        ov_model = OVKModel(model_dir, device)
        self.pipeline = KPipeline(lang_code="a", model=ov_model)

    def generate_audio(self, text: str, voice: str):
        st = time.perf_counter()
        with torch.no_grad():
            generator = self.pipeline(text, voice=voice)
            result = next(generator)
        latency = time.perf_counter() - st
        return result.audio, latency


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> io.BytesIO:        
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


if __name__ == "__main__":
    pipeline = TTS("Kokoro-82M","cpu")
    generator, r = pipeline.generate_audio("Hellow world how are you doing today?", voice="af_alloy")
    # print(generator)
    print(f"Audio mean: {torch.mean(generator)} Max: {torch.max(generator)} Min: {torch.min(generator)}")
    sf.write("test_wav.wav", generator, 24000)  # 24kHz sample rate
    print(f"Audio saved")