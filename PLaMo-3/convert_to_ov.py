#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
pfnet/plamo-3-nict-2b-base -> OpenVINO IR (stateful / KV-cache).

PLaMo-3 (`model_type = plamo3`, `Plamo3ForCausalLM`) is a decoder-only causal LM
shipped with custom modeling code (`trust_remote_code=True`) and a bespoke
sliding-window / full-attention cache, so it is NOT supported by optimum-intel's
`optimum-cli export openvino`. This script exports it directly as two graphs that
implement autoregressive decoding with an explicit key/value cache:

  * decoder (prefill) : input_ids [1,L] + attention_mask [1,L]
                          -> logits [1,L,V] + present.{i}.key / present.{i}.value
  * decoder_with_past : input_ids [1,1] + attention_mask [1,S+1]
                          + past.{i}.key / past.{i}.value
                          -> logits [1,1,V] + present.{i}.key / present.{i}.value

Per-layer self K/V shape: [1, num_kv_heads (4), seq, head_dim (128)], 24 layers.

Two OpenVINO tracing workarounds are applied only during export:
  1. `torch.isneginf` (an `_unmask_unattended` safety inside the attention) has
     no OpenVINO conversion rule -> replaced with an equality test against -inf.
  2. `DynamicLayer.lazy_initialization` seeds the empty cache with a rank-1
     tensor; the subsequent `torch.cat(dim=-2)` then fails in the OpenVINO
     frontend -> patched to a rank-4 zero-length tensor.

Usage:
    python convert_to_ov.py --output_dir ir --weight_format fp16
"""

import argparse
import contextlib
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("plamo3-convert")

MODEL_ID = "pfnet/plamo-3-nict-2b-base"
DECODER_IR = "openvino_decoder_model.xml"
DECODER_PAST_IR = "openvino_decoder_with_past_model.xml"
META_JSON = "ov_plamo3_kvcache.json"


# ---------------------------------------------------------------------------
# export-time patches
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def tracing_patches():
    """Patch torch.isneginf + DynamicLayer.lazy_initialization for export.

    The empty-cache patch is written to accept a variable number of arguments so
    it works across transformers versions (the signature of
    `lazy_initialization` changed from `(key_states)` to `(key_states,
    value_states)`).
    """
    from transformers.cache_utils import DynamicLayer

    orig_isneginf = torch.isneginf
    orig_lazy = DynamicLayer.lazy_initialization

    def patched_lazy(self, key_states, *args):
        self.dtype, self.device = key_states.dtype, key_states.device
        empty_shape = list(key_states.shape)
        empty_shape[-2] = 0  # rank-4 zero-length so cat(dim=-2) is valid in OV
        self.keys = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)

    torch.isneginf = lambda x: x == float("-inf")
    DynamicLayer.lazy_initialization = patched_lazy
    try:
        yield
    finally:
        torch.isneginf = orig_isneginf
        DynamicLayer.lazy_initialization = orig_lazy


# ---------------------------------------------------------------------------
# torch wrappers used for tracing
# ---------------------------------------------------------------------------
class PrefillWrapper(torch.nn.Module):
    """(input_ids, attention_mask) -> (logits, *present_kv)."""

    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.n = num_layers

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        cache = out.past_key_values
        outs = [out.logits]
        for i in range(self.n):
            outs.append(cache.layers[i].keys)
            outs.append(cache.layers[i].values)
        return tuple(outs)


class DecodeWrapper(torch.nn.Module):
    """(input_ids, attention_mask, *past_kv) -> (logits, *present_kv)."""

    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.n = num_layers

    def forward(self, input_ids, attention_mask, *past):
        from transformers import DynamicCache

        cache = DynamicCache()
        for i in range(self.n):
            cache.update(past[2 * i], past[2 * i + 1], i)
        # Plamo3Model converts a plain DynamicCache into its own Plamo3Cache.
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        new_cache = out.past_key_values
        outs = [out.logits]
        for i in range(self.n):
            outs.append(new_cache.layers[i].keys)
            outs.append(new_cache.layers[i].values)
        return tuple(outs)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def _name_decode_io(ov_model, num_layers):
    """Assign names + dynamic shapes to the *past varargs decode graph."""
    import openvino as ov

    ov_model.inputs[0].get_tensor().set_names({"input_ids"})
    ov_model.inputs[0].get_node().set_partial_shape(ov.PartialShape([1, -1]))
    ov_model.inputs[1].get_tensor().set_names({"attention_mask"})
    ov_model.inputs[1].get_node().set_partial_shape(ov.PartialShape([1, -1]))
    for i in range(num_layers):
        for j, kind in enumerate(("key", "value")):
            idx = 2 + 2 * i + j
            ov_model.inputs[idx].get_tensor().set_names({f"past.{i}.{kind}"})
            ov_model.inputs[idx].get_node().set_partial_shape(ov.PartialShape([1, -1, -1, -1]))
    ov_model.outputs[0].get_tensor().set_names({"logits"})
    for i in range(num_layers):
        for j, kind in enumerate(("key", "value")):
            ov_model.outputs[1 + 2 * i + j].get_tensor().set_names({f"present.{i}.{kind}"})
    ov_model.validate_nodes_and_infer_types()


def convert(output_dir: Path, weight_format: str):
    import openvino as ov
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s (downloads ~4GB on first run) ...", MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    log.info(
        "layers=%d attn_heads=%d kv_heads=%d head_dim=%d hidden=%d vocab=%d",
        num_layers, cfg.num_attention_heads, num_kv_heads, head_dim, cfg.hidden_size, cfg.vocab_size,
    )

    compress = weight_format == "fp16"

    with tracing_patches():
        # --- prefill graph ---
        log.info("Tracing + converting prefill graph ...")
        pf_ids = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.int64)
        pf_mask = torch.ones(1, 16, dtype=torch.int64)
        ov_prefill = ov.convert_model(
            PrefillWrapper(model, num_layers).eval(),
            example_input=(pf_ids, pf_mask),
            input=[
                ("input_ids", ov.PartialShape([1, -1]), ov.Type.i64),
                ("attention_mask", ov.PartialShape([1, -1]), ov.Type.i64),
            ],
        )
        ov_prefill.outputs[0].get_tensor().set_names({"logits"})
        for i in range(num_layers):
            for j, kind in enumerate(("key", "value")):
                ov_prefill.outputs[1 + 2 * i + j].get_tensor().set_names({f"present.{i}.{kind}"})
        ov.save_model(ov_prefill, output_dir / DECODER_IR, compress_to_fp16=compress)
        log.info("Saved prefill IR -> %s", output_dir / DECODER_IR)

        # --- decode-with-past graph ---
        log.info("Tracing + converting decode-with-past graph ...")
        past_len = 16
        dec_ids = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.int64)
        dec_mask = torch.ones(1, past_len + 1, dtype=torch.int64)
        past = []
        for _ in range(num_layers):
            past.append(torch.randn(1, num_kv_heads, past_len, head_dim, dtype=torch.float32))
            past.append(torch.randn(1, num_kv_heads, past_len, head_dim, dtype=torch.float32))
        ov_decode = ov.convert_model(
            DecodeWrapper(model, num_layers).eval(),
            example_input=(dec_ids, dec_mask, *past),
        )
        _name_decode_io(ov_decode, num_layers)
        ov.save_model(ov_decode, output_dir / DECODER_PAST_IR, compress_to_fp16=compress)
        log.info("Saved decode-with-past IR -> %s", output_dir / DECODER_PAST_IR)

    tokenizer.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)
    gc = model.generation_config
    meta = {
        "model_id": MODEL_ID,
        "num_layers": int(num_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "hidden_size": int(cfg.hidden_size),
        "intermediate_size": int(cfg.intermediate_size),
        "vocab_size": int(cfg.vocab_size),
        "window_size": int(cfg.window_size),
        "max_position_embeddings": int(cfg.max_position_embeddings),
        "eos_token_id": gc.eos_token_id if gc.eos_token_id is not None else cfg.eos_token_id,
        "pad_token_id": gc.pad_token_id if gc.pad_token_id is not None else cfg.pad_token_id,
        "bos_token_id": gc.bos_token_id if gc.bos_token_id is not None else cfg.bos_token_id,
        "decoder_ir": DECODER_IR,
        "decoder_with_past_ir": DECODER_PAST_IR,
    }
    (output_dir / META_JSON).write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata -> %s", output_dir / META_JSON)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Convert pfnet/plamo-3-nict-2b-base to OpenVINO IR (KV-cache)")
    ap.add_argument("--output_dir", default="ir", type=Path)
    ap.add_argument("--weight_format", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    out = args.output_dir
    if not out.is_absolute():
        out = Path(__file__).parent / out
    convert(out, args.weight_format)


if __name__ == "__main__":
    main()
