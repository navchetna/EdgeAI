import io
import numpy as np
import soundfile as sf
import torch
import json
from loguru import logger
import openvino as ov
from kokoro import KPipeline
from kokoro.model import KModel
from pathlib import Path

core = ov.Core()


class OVKModel(KModel):
    def __init__(self, model_dir: Path, device: str):
        torch.nn.Module.__init__(self)
        with (Path(model_dir) / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        self.vocab = config["vocab"]
        self.model = core.compile_model(Path(model_dir) / "openvino_model.xml", "GPU")
        self.context_length = config["plbert"]["max_position_embeddings"]


    @property
    def device(self):
        return torch.device("cpu")
    
    def forward_with_tokens(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1) -> tuple[torch.FloatTensor, torch.LongTensor]:
        outputs = self.model([input_ids, ref_s, torch.tensor(speed)])
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])
    

def get_tts_model(model_dir, device, model_id: str = "hexgrad/Kokoro-82M"):
    ov_model = OVKModel(model_dir, device)
    ov_pipeline = KPipeline(lang_code="a", repo_id=model_id, model=ov_model)
    logger.info("Warming up the model")
    _ = ov_pipeline("Hello this is a model testing pipeline")
    logger.info("Warmup done!")
    return ov_pipeline


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


# if __name__ == "__main__":
#     pipeline = get_tts_model("Kokoro-82M","cpu")
#     generator = next(pipeline("Hellow world", voice="af_alloy"))
#     print(generator)