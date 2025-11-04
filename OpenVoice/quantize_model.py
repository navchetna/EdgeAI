import os
import torch
import openvino as ov
from pathlib import Path

from openvoice.api import BaseSpeakerTTS, ToneColorConverter, OpenVoiceBaseClass
import nncf


CKPT_BASE_PATH = Path("model/checkpoints")


en_suffix = CKPT_BASE_PATH / "base_speakers/EN"
zh_suffix = CKPT_BASE_PATH / "base_speakers/ZH"
converter_suffix = CKPT_BASE_PATH / "converter"

enable_chinese_lang = False


class OVOpenVoiceBase(torch.nn.Module):
    """
    Base class for both TTS and voice tone conversion model: constructor is same for both of them.
    """

    def __init__(self, voice_model: OpenVoiceBaseClass):
        super().__init__()
        self.voice_model = voice_model
        for par in voice_model.model.parameters():
            par.requires_grad = False


class OVOpenVoiceTTS(OVOpenVoiceBase):
    """
    Constructor of this class accepts BaseSpeakerTTS object for speech generation and wraps it's 'infer' method with forward.
    """

    def get_example_input(self):
        stn_tst = self.voice_model.get_text("this is original text", self.voice_model.hps, False)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        speaker_id = torch.LongTensor([1])
        noise_scale = torch.tensor(0.667)
        length_scale = torch.tensor(1.0)
        noise_scale_w = torch.tensor(0.6)
        return (
            x_tst,
            x_tst_lengths,
            speaker_id,
            noise_scale,
            length_scale,
            noise_scale_w,
        )

    def forward(self, x, x_lengths, sid, noise_scale, length_scale, noise_scale_w):
        return self.voice_model.model.infer(x, x_lengths, sid, noise_scale, length_scale, noise_scale_w)


class OVOpenVoiceConverter(OVOpenVoiceBase):
    """
    Constructor of this class accepts ToneColorConverter object for voice tone conversion and wraps it's 'voice_conversion' method with forward.
    """

    def get_example_input(self):
        y = torch.randn([1, 513, 238], dtype=torch.float32)
        y_lengths = torch.LongTensor([y.size(-1)])
        target_se = torch.randn(*(1, 256, 1))
        source_se = torch.randn(*(1, 256, 1))
        tau = torch.tensor(0.3)
        return (y, y_lengths, source_se, target_se, tau)

    def forward(self, y, y_lengths, sid_src, sid_tgt, tau):
        return self.voice_model.model.voice_conversion(y, y_lengths, sid_src, sid_tgt, tau)



pt_device = "cpu"
core = ov.Core()

en_base_speaker_tts = BaseSpeakerTTS(en_suffix / "config.json", device=pt_device)
en_base_speaker_tts.load_ckpt(en_suffix / "checkpoint.pth")

tone_color_converter = ToneColorConverter(converter_suffix / "config.json", device=pt_device)
tone_color_converter.load_ckpt(converter_suffix / "checkpoint.pth")

if enable_chinese_lang:
    zh_base_speaker_tts = BaseSpeakerTTS(zh_suffix / "config.json", device=pt_device)
    zh_base_speaker_tts.load_ckpt(zh_suffix / "checkpoint.pth")
else:
    zh_base_speaker_tts = None


# Loading the model
IRS_PATH = Path("openvino_irs/")
EN_TTS_IR = IRS_PATH / "openvoice_en_tts.xml"
ZH_TTS_IR = IRS_PATH / "openvoice_zh_tts.xml"
VOICE_CONVERTER_IR = IRS_PATH / "openvoice_tone_conversion.xml"

paths = [EN_TTS_IR, VOICE_CONVERTER_IR]
models = [
    OVOpenVoiceTTS(en_base_speaker_tts),
    OVOpenVoiceConverter(tone_color_converter),
]
if enable_chinese_lang:
    models.append(OVOpenVoiceTTS(zh_base_speaker_tts))
    paths.append(ZH_TTS_IR)
ov_models = []

for model, path in zip(models, paths):
    if not path.exists():
        ov_model = ov.convert_model(model, example_input=model.get_example_input())
        ov_model = nncf.compress_weights(ov_model)
        ov.save_model(ov_model, path)
    else:
        ov_model = core.read_model(path)
    ov_models.append(ov_model)

ov_en_tts, ov_voice_conversion = ov_models[:2]
if enable_chinese_lang:
    ov_zh_tts = ov_models[-1]