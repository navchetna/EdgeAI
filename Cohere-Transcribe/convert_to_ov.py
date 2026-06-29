#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
CohereLabs/cohere-transcribe-03-2026 -> OpenVINO IR (stateful / KV-cache).

Exports three graphs that together implement autoregressive ASR decoding with an
explicit key/value cache (one token per decode step, cached self- and
cross-attention K/V):

  * encoder           : input_features [1,T,128] + attention_mask [1,T]
                          -> encoder_hidden_states [1,T',1280] + encoder_attention_mask [1,T']
  * decoder (prefill) : decoder_input_ids [1,L] + encoder outputs
                          -> logits [1,L,V] + present self/cross K/V (all layers)
  * decoder_with_past : decoder_input_ids [1,1] + encoder outputs + self_attention_mask
                          + past self/cross K/V
                          -> logits [1,1,V] + updated present self K/V (cross reused)

Self K/V grow by one each step; cross K/V are computed once at prefill and reused.

Usage:
    python convert_to_ov.py --output_dir ir --weight_format fp16
"""

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cohere-transcribe-convert")

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
SAMPLING_RATE = 16_000
ENCODER_IR = "openvino_encoder_model.xml"
DECODER_IR = "openvino_decoder_model.xml"
DECODER_PAST_IR = "openvino_decoder_with_past_model.xml"
META_JSON = "ov_cohere_transcribe_kvcache.json"


# ---------------------------------------------------------------------------
# export-time patch
# ---------------------------------------------------------------------------
def _patch_dynamic_layer_init() -> None:
    """Make DynamicLayer's empty cache rank-4 (zero-length) instead of rank-1.

    The default `lazy_initialization` uses `torch.tensor([])` (rank 1). Eager
    PyTorch special-cases `torch.cat` with such an empty tensor, but the OpenVINO
    torch frontend cannot convert that concat (invalid axis -2). Initialising the
    empty cache with the correct rank makes the prefill concat well-formed.
    """
    from transformers.cache_utils import DynamicLayer

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        empty_shape = list(key_states.shape)
        empty_shape[-2] = 0
        self.keys = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    DynamicLayer.lazy_initialization = lazy_initialization


# ---------------------------------------------------------------------------
# torch wrappers
# ---------------------------------------------------------------------------
class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features, attention_mask):
        enc = self.encoder(input_features, attention_mask=attention_mask)
        mask = enc.attention_mask
        if mask is None:
            mask = torch.ones(enc.last_hidden_state.shape[:2], dtype=torch.int32)
        return enc.last_hidden_state, mask.to(torch.int32)


class PrefillDecoder(torch.nn.Module):
    """No past -> logits + full self & cross K/V for every layer."""

    def __init__(self, model, num_layers):
        super().__init__()
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.num_layers = num_layers
        self._DynamicCache = DynamicCache
        self._EncoderDecoderCache = EncoderDecoderCache

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
        cache = self._EncoderDecoderCache(self._DynamicCache(), self._DynamicCache())
        out = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=torch.ones_like(decoder_input_ids),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        logits = self.proj_out(out.last_hidden_state)
        pkv = out.past_key_values
        outs = [logits]
        for i in range(self.num_layers):
            outs.append(pkv.self_attention_cache.layers[i].keys)
            outs.append(pkv.self_attention_cache.layers[i].values)
            outs.append(pkv.cross_attention_cache.layers[i].keys)
            outs.append(pkv.cross_attention_cache.layers[i].values)
        return tuple(outs)


class DecodeStep(torch.nn.Module):
    """Single token + past self/cross K/V -> logits + updated self K/V."""

    def __init__(self, model, num_layers):
        super().__init__()
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.num_layers = num_layers
        self._DynamicCache = DynamicCache
        self._EncoderDecoderCache = EncoderDecoderCache

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask, self_attention_mask, *past):
        sa = self._DynamicCache()
        ca = self._DynamicCache()
        cache = self._EncoderDecoderCache(sa, ca)
        for i in range(self.num_layers):
            sk, sv, ck, cv = past[4 * i : 4 * i + 4]
            sa.update(sk, sv, i)
            ca.update(ck, cv, i)
            cache.is_updated[i] = True  # reuse cross-attention K/V, skip recompute
        out = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=self_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        logits = self.proj_out(out.last_hidden_state)
        pkv = out.past_key_values
        outs = [logits]
        for i in range(self.num_layers):
            outs.append(pkv.self_attention_cache.layers[i].keys)
            outs.append(pkv.self_attention_cache.layers[i].values)
        return tuple(outs)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def convert(output_dir: Path, weight_format: str) -> dict:
    import openvino as ov
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    _patch_dynamic_layer_init()

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    enc_hidden = model.model.encoder.config.hidden_size
    vocab = cfg.vocab_size
    n_mels = getattr(cfg.encoder_config, "num_mel_bins", 128)
    log.info(
        "layers=%d kv_heads=%d head_dim=%d enc_hidden=%d vocab=%d",
        num_layers, num_kv_heads, head_dim, enc_hidden, vocab,
    )

    compress = weight_format == "fp16"

    # --- example tensors ---
    feat = torch.randn(1, 200, n_mels, dtype=torch.float32)
    amask = torch.ones(1, 200, dtype=torch.bool)
    with torch.no_grad():
        enc_out = model.model.encoder(feat, attention_mask=amask)
    ehs = enc_out.last_hidden_state
    emask = (enc_out.attention_mask if enc_out.attention_mask is not None else torch.ones(ehs.shape[:2])).to(torch.int32)

    # --- 1) encoder ---
    log.info("Converting encoder ...")
    ov_encoder = ov.convert_model(
        EncoderWrapper(model).eval(),
        example_input=(feat, amask),
        input=[
            ("input_features", ov.PartialShape([1, -1, n_mels]), ov.Type.f32),
            ("attention_mask", ov.PartialShape([1, -1]), ov.Type.boolean),
        ],
    )
    ov_encoder.outputs[0].get_tensor().set_names({"encoder_hidden_states"})
    ov_encoder.outputs[1].get_tensor().set_names({"encoder_attention_mask"})
    ov.save_model(ov_encoder, output_dir / ENCODER_IR, compress_to_fp16=compress)

    # --- 2) decoder prefill (no past) ---
    log.info("Converting decoder (prefill) ...")
    prompt = torch.zeros(1, 8, dtype=torch.int64)
    ov_prefill = ov.convert_model(
        PrefillDecoder(model, num_layers).eval(),
        example_input=(prompt, ehs, emask),
        input=[
            ("decoder_input_ids", ov.PartialShape([1, -1]), ov.Type.i64),
            ("encoder_hidden_states", ov.PartialShape([1, -1, enc_hidden]), ov.Type.f32),
            ("encoder_attention_mask", ov.PartialShape([1, -1]), ov.Type.i32),
        ],
    )
    ov_prefill.outputs[0].get_tensor().set_names({"logits"})
    idx = 1
    for i in range(num_layers):
        ov_prefill.outputs[idx + 0].get_tensor().set_names({f"present.{i}.self.key"})
        ov_prefill.outputs[idx + 1].get_tensor().set_names({f"present.{i}.self.value"})
        ov_prefill.outputs[idx + 2].get_tensor().set_names({f"present.{i}.cross.key"})
        ov_prefill.outputs[idx + 3].get_tensor().set_names({f"present.{i}.cross.value"})
        idx += 4
    ov.save_model(ov_prefill, output_dir / DECODER_IR, compress_to_fp16=compress)

    # --- 3) decoder with past (single token) ---
    log.info("Converting decoder_with_past ...")
    tok = torch.zeros(1, 1, dtype=torch.int64)
    past_len = 8
    self_mask = torch.ones(1, past_len + 1, dtype=torch.int64)
    past_self_k = torch.randn(1, num_kv_heads, past_len, head_dim)
    past_cross_k = torch.randn(1, num_kv_heads, emask.shape[1], head_dim)
    example_past = []
    for _ in range(num_layers):
        example_past += [past_self_k.clone(), past_self_k.clone(), past_cross_k.clone(), past_cross_k.clone()]

    # The decoder forward uses *past varargs, so the traced inputs don't carry our
    # names. Convert from the example, then assign names + dynamic shapes by index.
    ov_past = ov.convert_model(
        DecodeStep(model, num_layers).eval(),
        example_input=(tok, ehs, emask, self_mask, *example_past),
    )
    dyn4 = ov.PartialShape([1, num_kv_heads, -1, head_dim])
    in_specs = [
        ("decoder_input_ids", ov.PartialShape([1, 1])),
        ("encoder_hidden_states", ov.PartialShape([1, -1, enc_hidden])),
        ("encoder_attention_mask", ov.PartialShape([1, -1])),
        ("self_attention_mask", ov.PartialShape([1, -1])),
    ]
    for i in range(num_layers):
        in_specs.append((f"past.{i}.self.key", dyn4))
        in_specs.append((f"past.{i}.self.value", dyn4))
        in_specs.append((f"past.{i}.cross.key", dyn4))
        in_specs.append((f"past.{i}.cross.value", dyn4))
    assert len(in_specs) == len(ov_past.inputs), f"{len(in_specs)} specs vs {len(ov_past.inputs)} inputs"
    for inp, (name, shape) in zip(ov_past.inputs, in_specs):
        inp.get_tensor().set_names({name})
        inp.get_node().set_partial_shape(shape)
    ov_past.validate_nodes_and_infer_types()

    ov_past.outputs[0].get_tensor().set_names({"logits"})
    idx = 1
    for i in range(num_layers):
        ov_past.outputs[idx + 0].get_tensor().set_names({f"present.{i}.self.key"})
        ov_past.outputs[idx + 1].get_tensor().set_names({f"present.{i}.self.value"})
        idx += 2
    ov.save_model(ov_past, output_dir / DECODER_PAST_IR, compress_to_fp16=compress)

    # --- processor / metadata ---
    processor.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)
    gc = model.generation_config
    meta = {
        "model_id": MODEL_ID,
        "sampling_rate": SAMPLING_RATE,
        "n_mels": int(n_mels),
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "encoder_hidden": int(enc_hidden),
        "vocab_size": int(vocab),
        "eos_token_id": gc.eos_token_id,
        "pad_token_id": gc.pad_token_id,
        "bos_token_id": gc.bos_token_id,
        "encoder_ir": ENCODER_IR,
        "decoder_ir": DECODER_IR,
        "decoder_with_past_ir": DECODER_PAST_IR,
    }
    (output_dir / META_JSON).write_text(json.dumps(meta, indent=2))
    log.info("Saved stateful IR + metadata to %s", output_dir)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Convert cohere-transcribe to OpenVINO IR (KV-cache)")
    ap.add_argument("--output_dir", default="ir", type=Path, help="Directory for the OpenVINO IR files")
    ap.add_argument("--weight_format", choices=["fp16", "fp32"], default="fp16", help="Weight compression format")
    args = ap.parse_args()

    out = args.output_dir
    if not out.is_absolute():
        out = Path(__file__).parent / out

    convert(out, args.weight_format)


if __name__ == "__main__":
    main()
