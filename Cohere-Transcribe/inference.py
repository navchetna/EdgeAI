#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Inference for CohereLabs/cohere-transcribe-03-2026 on OpenVINO (stateful KV-cache).

Loads the three IR graphs produced by `convert_to_ov.py` (encoder, decoder
prefill, decoder_with_past) and runs greedy autoregressive transcription with an
external KV-cache: cross-attention K/V are computed once at prefill and reused;
self-attention K/V grow by one token per decode step.

Usage:
    python inference.py --model_dir ir --device CPU --audio sample.wav
    python inference.py --model_dir ir --device GPU            # uses a demo clip
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cohere-transcribe-infer")

SAMPLING_RATE = 16_000
META_JSON = "ov_cohere_transcribe_kvcache.json"


class KVCacheManager:
    """Holds self/cross K/V tensors between decode steps."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.self_kv: list[tuple[np.ndarray, np.ndarray]] = []
        self.cross_kv: list[tuple[np.ndarray, np.ndarray]] = []

    def init_from_prefill(self, prefill_out: dict) -> None:
        self.self_kv = [
            (prefill_out[f"present.{i}.self.key"], prefill_out[f"present.{i}.self.value"])
            for i in range(self.num_layers)
        ]
        self.cross_kv = [
            (prefill_out[f"present.{i}.cross.key"], prefill_out[f"present.{i}.cross.value"])
            for i in range(self.num_layers)
        ]

    @property
    def seq_len(self) -> int:
        return self.self_kv[0][0].shape[2] if self.self_kv else 0

    def as_decode_inputs(self) -> dict:
        feed = {}
        for i in range(self.num_layers):
            feed[f"past.{i}.self.key"] = self.self_kv[i][0]
            feed[f"past.{i}.self.value"] = self.self_kv[i][1]
            feed[f"past.{i}.cross.key"] = self.cross_kv[i][0]
            feed[f"past.{i}.cross.value"] = self.cross_kv[i][1]
        return feed

    def update_self(self, step_out: dict) -> None:
        self.self_kv = [
            (step_out[f"present.{i}.self.key"], step_out[f"present.{i}.self.value"]) for i in range(self.num_layers)
        ]


class CohereTranscribeOV:
    def __init__(self, model_dir: str = "ir", device: str = "CPU"):
        import openvino as ov
        from transformers import AutoProcessor

        model_dir = Path(model_dir)
        if not model_dir.is_absolute():
            model_dir = Path(__file__).parent / model_dir
        self.model_dir = model_dir

        self.meta = json.loads((model_dir / META_JSON).read_text())
        self.num_layers = self.meta["num_layers"]
        self.processor = AutoProcessor.from_pretrained(model_dir)

        eos = self.meta["eos_token_id"]
        self.eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}

        core = ov.Core()
        log.info("Compiling encoder / decoder / decoder_with_past for %s ...", device)
        self.encoder = core.compile_model(model_dir / self.meta["encoder_ir"], device)
        self.prefill = core.compile_model(model_dir / self.meta["decoder_ir"], device)
        self.decode = core.compile_model(model_dir / self.meta["decoder_with_past_ir"], device)

    @staticmethod
    def _np(x, dtype):
        return np.asarray(x.cpu() if hasattr(x, "cpu") else x).astype(dtype)

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 256) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLING_RATE, language="en", return_tensors="np")
        feat = self._np(inputs["input_features"], np.float32)
        amask = self._np(inputs["attention_mask"], bool)
        prompt = self._np(inputs["decoder_input_ids"], np.int64)

        enc_res = self.encoder({"input_features": feat, "attention_mask": amask})
        ehs = enc_res["encoder_hidden_states"]
        emask = enc_res["encoder_attention_mask"]

        # --- prefill ---
        pf = self.prefill(
            {"decoder_input_ids": prompt, "encoder_hidden_states": ehs, "encoder_attention_mask": emask}
        )
        pf = {k.get_any_name(): v for k, v in pf.items()}
        cache = KVCacheManager(self.num_layers)
        cache.init_from_prefill(pf)
        next_id = int(pf["logits"][0, -1].argmax())
        generated = [next_id]

        # --- decode loop ---
        for _ in range(max_new_tokens):
            if next_id in self.eos_set:
                break
            self_mask = np.ones((1, cache.seq_len + 1), dtype=np.int64)
            feed = {
                "decoder_input_ids": np.array([[next_id]], dtype=np.int64),
                "encoder_hidden_states": ehs,
                "encoder_attention_mask": emask,
                "self_attention_mask": self_mask,
            }
            feed.update(cache.as_decode_inputs())
            step = self.decode(feed)
            step = {k.get_any_name(): v for k, v in step.items()}
            cache.update_self(step)
            next_id = int(step["logits"][0, -1].argmax())
            generated.append(next_id)

        text = self.processor.batch_decode([generated], skip_special_tokens=True)[0]
        return text.strip()


def load_audio(audio_path: str | None) -> np.ndarray:
    if audio_path:
        import librosa

        audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
        return audio
    from datasets import load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    return ds[0]["audio"]["array"]


def main():
    ap = argparse.ArgumentParser(description="Run cohere-transcribe inference with OpenVINO (KV-cache)")
    ap.add_argument("--model_dir", default="ir", help="Directory with the converted OpenVINO IR")
    ap.add_argument("--device", default="CPU", help="OpenVINO device (CPU / GPU / NPU / AUTO)")
    ap.add_argument("--audio", default=None, help="Path to a .wav file; if omitted a demo clip is used")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    asr = CohereTranscribeOV(model_dir=args.model_dir, device=args.device)
    audio = load_audio(args.audio)

    st = time.perf_counter()
    text = asr.transcribe(audio, max_new_tokens=args.max_new_tokens)
    elapsed = time.perf_counter() - st

    log.info("=" * 60)
    log.info("Transcription:\n  %s", text)
    log.info("Latency: %.2f s", elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
