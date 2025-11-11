import os
import io
import numpy as np
import soundfile as sf
import torch
import openvino as ov
from pathlib import Path
from openvoice.api import BaseSpeakerTTS, ToneColorConverter, OpenVoiceBaseClass
import openvoice.se_extractor as se_extractor
import time



class OpenVoiceTTS():

    OUTPUT_DIR = Path("generated_audio")
    def __init__(
            self, 
            ov_irs_path: str = "openvino_irs", 
            ckpt_path: str = "model/checkpoints", 
            ref_audio_path: str = "resources/demo_speaker0.mp3",
            device: str = "cpu"
        ):

        DEVICE = "CPU"
        self.core = ov.Core()

        irs_path = Path(ov_irs_path)
        self.en_tts_ir = irs_path / "openvoice_en_tts.xml"
        self.zh_tts_ir = irs_path / "openvoice_zh_tts.xml"
        self.voice_converter_ir = irs_path / "openvoice_tone_conversion.xml"
        ckpt_base_path = Path(ckpt_path)
        self.en_suffix = ckpt_base_path / "base_speakers/EN"
        self.zh_suffix = ckpt_base_path / "base_speakers/ZH"
        self.converter_suffix = ckpt_base_path / "converter"


        self.en_source_default_se = torch.load(f"{self.en_suffix}/en_default_se.pth")
        self.en_source_style_se = torch.load(f"{self.en_suffix}/en_style_se.pth")
        # zh_source_se = torch.load(f"{self.zh_suffix}/zh_default_se.pth") 

        self.ov_en_tts = self.core.read_model(f"{ov_irs_path}\openvoice_en_tts.xml")
        self.ov_voice_conversion = self.core.read_model(f"{ov_irs_path}\openvoice_tone_conversion.xml")

        self.en_base_speaker_tts = BaseSpeakerTTS(self.en_suffix / "config.json", device=device)
        self.en_base_speaker_tts.load_ckpt(self.en_suffix / "checkpoint.pth")

        self.tone_color_converter = ToneColorConverter(self.converter_suffix / "config.json", device=device)
        self.tone_color_converter.load_ckpt(self.converter_suffix / "checkpoint.pth")


        self.en_base_speaker_tts.model.infer = self.get_pathched_infer(self.ov_en_tts, DEVICE)
        self.tone_color_converter.model.voice_conversion = self.get_patched_voice_conversion(self.ov_voice_conversion, DEVICE)

        output_path = "generated_audio"
        os.makedirs(output_path, exist_ok=True)
        self.target_se, self.audio_name = se_extractor.get_se(ref_audio_path, self.tone_color_converter, target_dir=output_path, vad=True)


    def generate_cloned_voice(self, text: str, output_path: str):
        audio, latency = self.generate_audio(text=text)
        self.tone_color_converter.convert(
            audio_src_path=audio,
            src_se=self.en_source_default_se,
            tgt_se=self.target_se,
            output_path=output_path,
            tau=0.3,
            message="@MyShell",
        )

        return f"Audio saved at: {output_path}"


    def generate_audio(self, text: str, voice: str = "default", language: str = "English", speed: float = 0.8):
        st = time.perf_counter()
        audio = self.en_base_speaker_tts.tts(text, None, speaker=voice, language=language, speed=speed)
        latency = time.perf_counter() - st
        return audio, latency



    def get_pathched_infer(self, ov_model: ov.Model, device: str) -> callable:
        compiled_model = self.core.compile_model(ov_model, device)

        def infer_impl(x, x_lengths, sid, noise_scale, length_scale, noise_scale_w):
            ov_output = compiled_model((x, x_lengths, sid, noise_scale, length_scale, noise_scale_w))
            return (torch.tensor(ov_output[0]),)

        return infer_impl


    def get_patched_voice_conversion(self, ov_model: ov.Model, device: str) -> callable:
        compiled_model = self.core.compile_model(ov_model, device)

        def voice_conversion_impl(y, y_lengths, sid_src, sid_tgt, tau):
            ov_output = compiled_model((y, y_lengths, sid_src, sid_tgt, tau))
            return (torch.tensor(ov_output[0]),)

        return voice_conversion_impl
    

def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> io.BytesIO:        
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


# if __name__ == "__main__":
#     import time
#     tts = OpenVoiceTTS()
#     text = "Hello world how are you doing today. I am doing really good. Let me know what you think about me today?"
#     st = time.time()
#     # audio = tts.generate_audio(text=text)
#     path = Path("generated_audio") / "tmp.wav"
#     tts.generate_cloned_voice(text, orig_voice_path=path)
#     print("Time: ", time.time() - st)