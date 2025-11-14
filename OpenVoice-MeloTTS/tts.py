import os
import io
import numpy as np
import soundfile as sf
import torch
import openvino as ov
from pathlib import Path
import time
import sys

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(repo_root, "OpenVoice"))
sys.path.append(os.path.join(repo_root, "MeloTTS"))

from melo.api import TTS
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import openvoice.se_extractor as se_extractor



class OpenVoiceTTS():

    OUTPUT_DIR = Path("generated_audio")
    def __init__(
            self, 
            ov_irs_path: str = "openvino_irs", 
            ckpt_path: str = "checkpoints", 
            ref_audio_path: str = "demo_voice.wav",
            device: str = "cpu",
        ):

        DEVICE = "CPU"
        self.core = ov.Core()

        irs_path = Path(ov_irs_path)
        self.en_tts_ir = irs_path / "melo_tts_en_newest.xml"
        self.voice_converter_ir = irs_path / "openvoice2_tone_conversion.xml"
        
        ckpt_base_path = Path(ckpt_path)

        self.en_suffix = ckpt_base_path / "base_speakers" / "ses"
        self.converter_suffix = ckpt_base_path / "converter"


        self.en_source_default_se = torch.load(f"{self.en_suffix}/en-india.pth")

        self.ov_en_tts = self.core.read_model(self.en_tts_ir)
        self.ov_voice_conversion = self.core.read_model(self.voice_converter_ir)

        self.melo_tts_en_newest = TTS(
                                    "EN_NEWEST",
                                    device,
                                    use_hf=False,
                                    config_path=ckpt_base_path / "MeloTTS-English-v3" /"config.json",
                                    ckpt_path=ckpt_base_path / "MeloTTS-English-v3" / "checkpoint.pth",
                                )                    
        self.voice = self.melo_tts_en_newest.hps.data.spk2id

        self.tone_color_converter = ToneColorConverter(self.converter_suffix / "config.json", device=device)
        self.tone_color_converter.load_ckpt(self.converter_suffix / "checkpoint.pth")


        self.melo_tts_en_newest.model.infer = self.get_pathched_infer(self.ov_en_tts, DEVICE)
        self.tone_color_converter.model.voice_conversion = self.get_patched_voice_conversion(self.ov_voice_conversion, DEVICE)


        output_path = "generated_audio"
        os.makedirs(output_path, exist_ok=True)
        self.target_se, self.audio_name = se_extractor.get_se(ref_audio_path, self.tone_color_converter, target_dir=output_path, vad=True)


    def generate_cloned_voice(self, text: str):
        st = time.perf_counter()
        audio, _ = self.generate_audio(text=text)
        audio = self.tone_color_converter.convert(
            audio_src_path=audio,
            src_se=self.en_source_default_se,
            tgt_se=self.target_se,
            output_path=None,
            tau=0.3,
            message="@MyShell",
        )
        latency = time.perf_counter() - st
        return audio, latency


    def generate_audio(self, text: str, voice: str = "EN_INDIA", speed: float = 0.8):
        st = time.perf_counter()
        audio = self.melo_tts_en_newest.tts_to_file(text, 0, None, speed=speed)
        latency = time.perf_counter() - st
        return audio, latency



    def get_pathched_infer(self, ov_model: ov.Model, device: str) -> callable:
        compiled_model = self.core.compile_model(ov_model, device)

        def infer_impl(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            ja_bert,
            noise_scale,
            length_scale,
            noise_scale_w,
            max_len=None,
            sdp_ratio=1.0,
            y=None,
            g=None,
        ):
            ov_output = compiled_model(
                (
                    x,
                    x_lengths,
                    sid,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    noise_scale,
                    length_scale,
                    noise_scale_w,
                    sdp_ratio,
                )
            )

            return (torch.tensor(ov_output[0]),)

        return infer_impl


    def get_patched_voice_conversion(self, ov_model: ov.Model, device: str) -> callable:
        compiled_model = self.core.compile_model(ov_model, device)

        def voice_conversion_impl(y, y_lengths, sid_src, sid_tgt, tau):
            ov_output = compiled_model((y, y_lengths, sid_src, sid_tgt, tau))
            return (torch.tensor(ov_output[0]),)

        return voice_conversion_impl

        

def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 44100) -> io.BytesIO:        
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


if __name__ == "__main__":
    import time
    tts = OpenVoiceTTS()
    text = "Hello world how are you doing today. I am doing really good. Let me know what you think about me today?"
    st = time.time()
    audio, tt = tts.generate_audio(text=text)
    print(tt)
    # path = Path("generated_audio") / "tmp.wav"
    # tts.generate_cloned_voice(text, output_path=path)
    # print("Time: ", time.time() - st)