#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Inference + throughput benchmark for pfnet/plamo-3-nict-2b-base on OpenVINO.

Loads the two IR graphs produced by `convert_to_ov.py` (decoder prefill and
decoder_with_past) and runs greedy autoregressive generation with an external
KV-cache: self-attention K/V grow by one token per decode step.

The benchmark generates a fixed number of tokens per run (early EOS is ignored so
throughput is comparable across runs), executes 3 warmup runs followed by 3
timed runs, and reports decode tokens/second (mean of the 3 timed runs).

Usage:
    python inference.py --model_dir ir --device CPU
    python inference.py --model_dir ir --device GPU --prompt "The capital of Japan is"
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("plamo3-infer")

META_JSON = "ov_plamo3_kvcache.json"

WARMUP_RUNS = 3
TIMED_RUNS = 3


class KVCacheManager:
    """Holds per-layer self K/V tensors between decode steps."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.kv = [None] * (2 * num_layers)

    def update(self, out: dict) -> None:
        for i in range(self.num_layers):
            self.kv[2 * i] = out[f"present.{i}.key"]
            self.kv[2 * i + 1] = out[f"present.{i}.value"]

    @property
    def seq_len(self) -> int:
        return self.kv[0].shape[2] if self.kv[0] is not None else 0

    def as_past_inputs(self) -> dict:
        feed = {}
        for i in range(self.num_layers):
            feed[f"past.{i}.key"] = self.kv[2 * i]
            feed[f"past.{i}.value"] = self.kv[2 * i + 1]
        return feed


class Plamo3OV:
    def __init__(self, model_dir: str = "ir", device: str = "CPU"):
        import openvino as ov
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        if not model_dir.is_absolute():
            model_dir = Path(__file__).parent / model_dir
        self.model_dir = model_dir

        self.meta = json.loads((model_dir / META_JSON).read_text())
        self.num_layers = self.meta["num_layers"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        eos = self.meta["eos_token_id"]
        self.eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}

        core = ov.Core()
        log.info("Compiling prefill / decode graphs for %s ...", device)
        self.prefill = core.compile_model(model_dir / self.meta["decoder_ir"], device)
        self.decode = core.compile_model(model_dir / self.meta["decoder_with_past_ir"], device)

    def generate(self, prompt: str, max_new_tokens: int = 128, ignore_eos: bool = False):
        """Greedy generation. Returns (text, num_new_tokens, prefill_s, decode_s)."""
        ids = self.tokenizer(prompt, return_tensors="np").input_ids.astype(np.int64)
        prompt_len = ids.shape[1]

        # --- prefill ---
        t0 = time.perf_counter()
        pf = self.prefill({"input_ids": ids, "attention_mask": np.ones((1, prompt_len), dtype=np.int64)})
        pf = {self.prefill.output(i).any_name: pf[i] for i in range(len(pf))}
        prefill_s = time.perf_counter() - t0

        cache = KVCacheManager(self.num_layers)
        cache.update(pf)
        next_id = int(pf["logits"][0, -1].argmax())

        # --- decode loop ---
        generated = []
        decode_s = 0.0
        for _ in range(max_new_tokens):
            if next_id in self.eos_set and not ignore_eos:
                break
            generated.append(next_id)
            feed = {
                "input_ids": np.array([[next_id]], dtype=np.int64),
                "attention_mask": np.ones((1, cache.seq_len + 1), dtype=np.int64),
            }
            feed.update(cache.as_past_inputs())
            t0 = time.perf_counter()
            step = self.decode(feed)
            decode_s += time.perf_counter() - t0
            step = {self.decode.output(i).any_name: step[i] for i in range(len(step))}
            cache.update(step)
            next_id = int(step["logits"][0, -1].argmax())

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, len(generated), prefill_s, decode_s


def main():
    ap = argparse.ArgumentParser(description="PLaMo-3 OpenVINO inference + throughput benchmark")
    ap.add_argument("--model_dir", default="ir", help="Directory with the converted OpenVINO IR")
    ap.add_argument("--device", default="CPU", help="OpenVINO device: CPU / GPU / NPU / AUTO")
    ap.add_argument("--prompt", default="The capital of Japan is", help="Prompt text")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="Tokens to generate per run")
    args = ap.parse_args()

    llm = Plamo3OV(model_dir=args.model_dir, device=args.device)

    # show one generation for a sanity check
    text, n_tok, _, _ = llm.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    log.info("Prompt    : %s", args.prompt)
    log.info("Generated : %s", text)

    # --- benchmark: fixed token count, ignore EOS for stable throughput ---
    log.info("=" * 60)
    log.info("Warmup: %d run(s) | Timed: %d run(s) | tokens/run: %d | device: %s",
             WARMUP_RUNS, TIMED_RUNS, args.max_new_tokens, args.device)

    for _ in range(WARMUP_RUNS):
        llm.generate(args.prompt, max_new_tokens=args.max_new_tokens, ignore_eos=True)

    tok_per_s = []
    for r in range(TIMED_RUNS):
        _, n_tok, prefill_s, decode_s = llm.generate(
            args.prompt, max_new_tokens=args.max_new_tokens, ignore_eos=True
        )
        tps = n_tok / decode_s if decode_s > 0 else 0.0
        tok_per_s.append(tps)
        log.info("Run %d: %d tokens | prefill %.3f s | decode %.3f s | %.2f tok/s",
                 r + 1, n_tok, prefill_s, decode_s, tps)

    mean = float(np.mean(tok_per_s))
    std = float(np.std(tok_per_s))
    log.info("-" * 60)
    log.info("Mean decode throughput: %.2f tok/s  (+/- %.2f over %d runs)", mean, std, TIMED_RUNS)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
