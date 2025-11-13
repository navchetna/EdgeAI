import os
import torch
import sys
import openvino as ov
from pathlib import Path

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(repo_root, "OpenVoice"))
sys.path.append(os.path.join(repo_root, "MeloTTS"))

from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
from melo.api import TTS
import nncf


CKPT_BASE_PATH = Path("checkpoints")


base_speakers_suffix = CKPT_BASE_PATH / "base_speakers" / "ses"
converter_suffix = CKPT_BASE_PATH / "converter"

melotts_english_suffix = CKPT_BASE_PATH / "MeloTTS-English-v3"


class OVSynthesizerTTSWrapper(torch.nn.Module):
    """
    Wrapper for SynthesizerTrn model from MeloTTS to make it compatible with Torch-style inference.
    """

    def __init__(self, model, language):
        super().__init__()
        self.model = model
        self.language = language

    def forward(
        self,
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
    ):
        """
        Forward call to the underlying SynthesizerTrn model. Accepts arbitrary arguments
        and forwards them directly to the model's inference method.
        """
        return self.model.infer(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            ja_bert,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )

    def get_example_input(self):
        """
        Return a tuple of example inputs for tracing/ONNX exporting or debugging.
        When exporting the SynthesizerTrn function,
        This model has been found to be very sensitive to the example_input used for model transformation.
        Here, we have implemented some simple rules or considered using real input data.
        """

        def gen_interleaved_random_tensor(length, value_range):
            """Generate a Tensor in the format [0, val, 0, val, ..., 0], val ∈ [low, high)."""
            return torch.tensor([[0 if i % 2 == 0 else torch.randint(*value_range, (1,)).item() for i in range(length)]], dtype=torch.int64).to(pt_device)

        def gen_interleaved_fixed_tensor(length, fixed_value):
            """Generate a Tensor in the format [0, val, 0, val, ..., 0]"""
            interleaved = [0 if i % 2 == 0 else fixed_value for i in range(length)]
            return torch.tensor([interleaved], dtype=torch.int64).to(pt_device)

        if self.language == "EN_NEWEST":
            seq_len = 73
            x_tst = gen_interleaved_random_tensor(seq_len, (14, 220))
            x_tst[:3] = 0
            x_tst[-3:] = 0
            x_tst_lengths = torch.tensor([seq_len], dtype=torch.int64).to(pt_device)
            speakers = torch.tensor([0], dtype=torch.int64).to(pt_device)  # This model has only one fixed id for speakers.
            tones = gen_interleaved_random_tensor(seq_len, (5, 10))
            lang_ids = gen_interleaved_fixed_tensor(seq_len, 2)  # lang_id for english
            bert = torch.randn((1, 1024, seq_len), dtype=torch.float32).to(pt_device)
            ja_bert = torch.randn(1, 768, seq_len, dtype=torch.float32).to(pt_device)
            sdp_ratio = torch.tensor(0.2).to(pt_device)
            noise_scale = torch.tensor(0.6).to(pt_device)
            noise_scale_w = torch.tensor(0.8).to(pt_device)
            length_scale = torch.tensor(1.0).to(pt_device)
        elif self.language == "ZH":
            seq_len = 37
            x_tst = gen_interleaved_random_tensor(seq_len, (7, 100))
            x_tst[:3] = 0
            x_tst[-3:] = 0
            x_tst_lengths = torch.tensor([37], dtype=torch.int64).to(pt_device)
            speakers = torch.tensor([1], dtype=torch.int64).to(pt_device)  # This model has only one fixed id for speakers.
            tones = gen_interleaved_random_tensor(seq_len, (4, 9))
            lang_ids = gen_interleaved_fixed_tensor(seq_len, 3)  # lang_id for chinese
            bert = torch.zeros((1, 1024, 37), dtype=torch.float32).to(pt_device)
            ja_bert = torch.randn(1, 768, 37).float().to(pt_device)
            sdp_ratio = torch.tensor(0.2).to(pt_device)
            noise_scale = torch.tensor(0.6).to(pt_device)
            noise_scale_w = torch.tensor(0.8).to(pt_device)
            length_scale = torch.tensor(1.0).to(pt_device)
        return (
            x_tst,
            x_tst_lengths,
            speakers,
            tones,
            lang_ids,
            bert,
            ja_bert,
            noise_scale,
            length_scale,
            noise_scale_w,
            sdp_ratio,
        )


class OVOpenVoiceConverter(torch.nn.Module):
    def __init__(self, voice_model: OpenVoiceBaseClass):
        super().__init__()
        self.voice_model = voice_model
        for par in voice_model.model.parameters():
            par.requires_grad = False

    def get_example_input(self):
        y = torch.randn([1, 513, 238], dtype=torch.float32)
        y_lengths = torch.LongTensor([y.size(-1)])
        target_se = torch.randn(*(1, 256, 1))
        source_se = torch.randn(*(1, 256, 1))
        tau = torch.tensor(0.3)
        return (y, y_lengths, source_se, target_se, tau)

    def forward(self, y, y_lengths, sid_src, sid_tgt, tau):
        """
        wraps the 'voice_conversion' method with forward.
        """
        return self.voice_model.model.voice_conversion(y, y_lengths, sid_src, sid_tgt, tau)


pt_device = "cpu"
core = ov.Core()

melo_tts_en_newest = TTS(
    "EN_NEWEST",
    pt_device,
    use_hf=False,
    config_path=melotts_english_suffix / "config.json",
    ckpt_path=melotts_english_suffix / "checkpoint.pth",
)

tone_color_converter = ToneColorConverter(converter_suffix / "config.json", device=pt_device)
tone_color_converter.load_ckpt(converter_suffix / "checkpoint.pth")
print(f"ToneColorConverter version: {tone_color_converter.version}")


# Loading the model
IRS_PATH = Path("openvino_irs/")
EN_TTS_IR = IRS_PATH / "melo_tts_en_newest.xml"
VOICE_CONVERTER_IR = IRS_PATH / "openvoice2_tone_conversion.xml"

paths = [EN_TTS_IR, VOICE_CONVERTER_IR]
models = [
    OVSynthesizerTTSWrapper(melo_tts_en_newest.model, "EN_NEWEST"),
    OVOpenVoiceConverter(tone_color_converter),
]

ov_models = []

for model, path in zip(models, paths):
    if not path.exists():
        ov_model = ov.convert_model(model, example_input=model.get_example_input())
        ov_model = nncf.compress_weights(ov_model)
        ov.save_model(ov_model, path)
    else:
        ov_model = core.read_model(path)
    ov_models.append(ov_model)

ov_en_tts, ov_zh_tts, ov_voice_conversion = ov_models