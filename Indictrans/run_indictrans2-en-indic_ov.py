"""
OpenVINO inference class for IndicTrans2 models - Fixed KV-Cache Implementation.

This module provides an OpenVINO-based inference engine with correct KV-cache handling
for seq2seq models. The key fix is proper management of self-attention KV-cache
which grows with each generation step.

Usage:
    from indictrans2_openvino_fixed import IndicTrans2OpenVINO
    
    translator = IndicTrans2OpenVINO(
        model_dir="./openvino_models/indictrans2-en-indic-1B-fp32/optimum",
        device="GPU"
    )
    
    translations = translator.translate(
        ["My name is ritik and I like in Bangalore?"],
        src_lang="eng_Latn",
        tgt_lang="hin_Deva"
    )
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndicTrans2OpenVINO:
    """
    OpenVINO-based inference engine for IndicTrans2 with correct KV-cache.
    
    Args:
        model_dir: Path to directory containing converted OpenVINO models
        device: Target device for inference ("CPU", "GPU", "AUTO")
        max_length: Maximum sequence length for generation
        num_beams: Number of beams for beam search (currently only supports 1)
    """
    
    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "GPU",
        max_length: int = 256,
        num_beams: int = 1,
        model_name: Optional[str] = None,
        tokenizer = None
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
        
        try:
            import openvino as ov
            self.ov = ov
            logger.info(f"OpenVINO version: {ov.__version__}")
        except ImportError:
            raise ImportError("OpenVINO is not installed. Install with: pip install openvino")
        
        try:
            from IndicTransToolkit.processor import IndicProcessor
            self.ip = IndicProcessor(inference=True)
            logger.info("IndicProcessor loaded")
        except ImportError:
            raise ImportError("IndicTransToolkit is not installed. Install with: pip install IndicTransToolkit")
        
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            
            # Auto-detect model type from directory or use provided name
            source = model_name if model_name else "ai4bharat/indictrans2-en-indic-1B"
            
            try:
                logger.info(f"Loading tokenizer from: {source}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    source,
                    trust_remote_code=True,
                    local_files_only=not source.startswith("ai4bharat/")
                )
                logger.info(f"Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                raise RuntimeError(f"Could not load tokenizer from {source}")
        
        self.config = self._load_config()
        self._load_models()
        
        logger.info(f"IndicTrans2OpenVINO initialized with device: {device}")
    
    def _load_config(self) -> dict:
        """Load model configuration."""
        config_path = self.model_dir / "openvino_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    def _load_models(self):
        """Load and compile OpenVINO models."""
        core = self.ov.Core()
        
        # Load encoder
        encoder_path = self.model_dir / "encoder.xml"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
        
        logger.info(f"Loading encoder from: {encoder_path}")
        self.encoder = core.compile_model(core.read_model(encoder_path), self.device)
        logger.info("  ✓ Encoder compiled")
        
        # Load decoder prefill (first step)
        decoder_prefill_path = self.model_dir / "decoder_prefill.xml"
        if not decoder_prefill_path.exists():
            raise FileNotFoundError(f"Decoder prefill model not found: {decoder_prefill_path}")
        
        logger.info(f"Loading decoder prefill from: {decoder_prefill_path}")
        self.decoder_prefill = core.compile_model(core.read_model(decoder_prefill_path), self.device)
        logger.info("  ✓ Decoder prefill compiled")
        
        # Load decoder decode (subsequent steps with KV-cache)
        decoder_decode_path = self.model_dir / "decoder_decode.xml"
        if not decoder_decode_path.exists():
            raise FileNotFoundError(f"Decoder decode model not found: {decoder_decode_path}")
        
        logger.info(f"Loading decoder decode from: {decoder_decode_path}")
        self.decoder_decode = core.compile_model(core.read_model(decoder_decode_path), self.device)
        logger.info("  ✓ Decoder decode compiled")
        
        # Load model config
        model_config = self.config.get("model_config", {})
        self.num_layers = model_config.get("decoder_layers", 18)
        self.num_heads = model_config.get("decoder_attention_heads", 16)
        self.embed_dim = model_config.get("encoder_embed_dim", 1024)
        self.head_dim = self.embed_dim // self.num_heads
        self.vocab_size = model_config.get("decoder_vocab_size", 256000)
        self.pad_token_id = model_config.get("pad_token_id", 1)
        self.eos_token_id = model_config.get("eos_token_id", 2)
        self.bos_token_id = model_config.get("bos_token_id", 0)
        self.decoder_start_token_id = model_config.get("decoder_start_token_id", 2)
        
        logger.info(f"Model config: layers={self.num_layers}, heads={self.num_heads}, "
                   f"embed_dim={self.embed_dim}, vocab_size={self.vocab_size}")
    
    def preprocess(
        self,
        sentences: List[str],
        src_lang: str,
        tgt_lang: str
    ) -> Dict[str, np.ndarray]:
        """Preprocess input sentences for translation."""
        # Apply IndicProcessor preprocessing
        preprocessed = self.ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Tokenize
        inputs = self.tokenizer(
            preprocessed,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True
        )
        
        return {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
    
    def postprocess(
        self,
        token_ids: np.ndarray,
        tgt_lang: str
    ) -> List[str]:
        """Postprocess generated token IDs to text."""
        # Filter out special tokens and padding
        cleaned_ids = []
        for seq in token_ids:
            # Remove padding and EOS tokens
            seq = seq[seq != self.pad_token_id]
            seq = seq[seq != self.eos_token_id]
            cleaned_ids.append(seq.tolist())
        
        decoded = self.tokenizer.batch_decode(
            cleaned_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return self.ip.postprocess_batch(decoded, lang=tgt_lang)
    
    def encode(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """Run encoder to get encoder hidden states."""
        return self.encoder({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })[0]
    
    def decode_prefill(
        self,
        decoder_input_ids: np.ndarray,
        encoder_hidden_states: np.ndarray,
        encoder_attention_mask: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, ...]]]:
        """
        Run decoder prefill step (first token generation).
        
        Returns:
            logits: (batch_size, 1, vocab_size)
            past_key_values: List of 4 tensors per layer (self_k, self_v, cross_k, cross_v)
        """
        results = self.decoder_prefill([
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask
        ])
        
        logits = results[0]
        past_key_values = []
        
        # Extract KV-cache: 4 tensors per layer (self_attn_k, self_attn_v, cross_attn_k, cross_attn_v)
        for i in range(self.num_layers):
            idx = 1 + i * 4
            past_key_values.append((
                results[idx],      # self_attn_k: (batch, heads, 1, head_dim)
                results[idx + 1],  # self_attn_v: (batch, heads, 1, head_dim)
                results[idx + 2],  # cross_attn_k: (batch, heads, enc_seq_len, head_dim)
                results[idx + 3]   # cross_attn_v: (batch, heads, enc_seq_len, head_dim)
            ))
        
        return logits, past_key_values
    
    def decode_step(
        self,
        decoder_input_ids: np.ndarray,
        encoder_hidden_states: np.ndarray,
        encoder_attention_mask: np.ndarray,
        past_key_values: List[Tuple[np.ndarray, ...]]
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, ...]]]:
        """
        Run single decoder step with KV-cache.
        
        CRITICAL FIX: Properly handle growing self-attention KV-cache.
        """
        # Build inputs: decoder_input_ids, encoder_hidden_states, encoder_attention_mask, then all KV tensors
        inputs = [
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask,
        ]
        
        # Add all KV tensors - order matters! (self_k, self_v, cross_k, cross_v) per layer
        for self_k, self_v, cross_k, cross_v in past_key_values:
            inputs.extend([self_k, self_v, cross_k, cross_v])
        
        results = self.decoder_decode(inputs)
        logits = results[0]
        
        new_past_key_values = []
        for i in range(self.num_layers):
            idx = 1 + i * 4
            new_past_key_values.append((
                results[idx],      # updated self_attn_k (now includes current step)
                results[idx + 1],  # updated self_attn_v
                results[idx + 2],  # cross_attn_k (unchanged)
                results[idx + 3]   # cross_attn_v (unchanged)
            ))
        
        return logits, new_past_key_values
    
    def generate(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate translations using autoregressive decoding with proper KV-cache.
        """
        max_length = max_length or self.max_length
        num_beams = num_beams or self.num_beams
        
        if num_beams != 1:
            logger.warning("Only num_beams=1 is supported in this implementation")
        
        batch_size = input_ids.shape[0]
        
        # Encode input
        encoder_hidden_states = self.encode(input_ids, attention_mask)
        enc_seq_len = encoder_hidden_states.shape[1]
        
        # Initialize decoder with BOS token
        decoder_input_ids = np.full((batch_size, 1), self.decoder_start_token_id, dtype=np.int64)
        
        # Prefill step: process BOS token, get initial KV-cache
        logits, past_key_values = self.decode_prefill(
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask
        )
        
        # Get first token
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        next_tokens = np.argmax(next_token_logits, axis=-1, keepdims=True)  # (batch_size, 1)
        
        # Track finished sequences
        finished = (next_tokens[:, 0] == self.eos_token_id)
        active_mask = ~finished
        
        generated_tokens = [next_tokens]
        
        # Autoregressive generation
        for step in range(1, max_length):
            if not np.any(active_mask):
                break
            
            # Only process active sequences
            # For simplicity, we process whole batch but mask out finished sequences
            
            logits, past_key_values = self.decode_step(
                next_tokens,  # Last generated token(s)
                encoder_hidden_states,
                attention_mask,
                past_key_values
            )
            
            next_token_logits = logits[:, -1, :]
            next_tokens = np.argmax(next_token_logits, axis=-1, keepdims=True)
            
            # Update finished mask
            just_finished = (next_tokens[:, 0] == self.eos_token_id) & active_mask
            active_mask &= ~just_finished
            
            generated_tokens.append(next_tokens.copy())
        
        return np.concatenate(generated_tokens, axis=1)
    
    def translate(
        self,
        sentences: List[str],
        src_lang: str,
        tgt_lang: str,
        max_length: Optional[int] = None
    ) -> List[str]:
        """Translate a list of sentences."""
        inputs = self.preprocess(sentences, src_lang, tgt_lang)
        
        output_ids = self.generate(
            inputs["input_ids"],
            inputs["attention_mask"],
            max_length=max_length
        )
        
        return self.postprocess(output_ids, tgt_lang)
    
    def batch_translate(
        self,
        input_sentences: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 32
    ) -> Dict[int, dict]:
        """
        Translate sentences in batches with profiling.
        """
        benchmark_data = {}
        
        for i in range(0, len(input_sentences), batch_size):
            batch = input_sentences[i:i+batch_size]
            
            t0 = time.perf_counter()
            inputs = self.preprocess(batch, src_lang, tgt_lang)
            preprocess_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            output_ids = self.generate(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
            generate_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            translations = self.postprocess(output_ids, tgt_lang)
            postprocess_time = time.perf_counter() - t0
            
            input_tokens = (inputs["input_ids"] != self.pad_token_id).sum(axis=1).tolist()
            output_tokens = []
            for seq in output_ids:
                # Count tokens until EOS or end
                eos_positions = np.where(seq == self.eos_token_id)[0]
                if len(eos_positions) > 0:
                    output_tokens.append(eos_positions[0] + 1)  # Include EOS
                else:
                    output_tokens.append(len(seq))
            
            print(f"Batch {i // batch_size + 1}: "
                  f"Preprocess={preprocess_time:.3f}s, "
                  f"Generate={generate_time:.3f}s, "
                  f"Postprocess={postprocess_time:.3f}s")
            
            benchmark_data[i // batch_size] = {
                "input_sentences": batch,
                "translations": translations,
                "times": {
                    "preprocess": preprocess_time,
                    "generate": generate_time,
                    "postprocess": postprocess_time,
                    "total": preprocess_time + generate_time + postprocess_time
                },
                "tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_input_tokens": sum(input_tokens),
                    "total_output_tokens": sum(output_tokens)
                }
            }
        
        return benchmark_data
    
    def warmup(self, num_iterations: int = 3):
        """Warm up the model by running dummy inferences."""
        logger.info("Warming up OpenVINO models...")
        dummy_sentences = ["This is a test sentence."] * 2
        
        for _ in range(num_iterations):
            _ = self.translate(dummy_sentences, "eng_Latn", "hin_Deva")
        
        logger.info("Warmup complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IndicTrans2 OpenVINO Inference (Fixed)")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to converted OpenVINO models")
    parser.add_argument("--device", type=str, default="GPU", choices=["CPU", "GPU", "AUTO"])
    parser.add_argument("--src-lang", type=str, default="eng_Latn")
    parser.add_argument("--tgt-lang", type=str, default="hin_Deva")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model-name", type=str, default=None, help="HuggingFace model name for tokenizer")
    
    args = parser.parse_args()
    
    # Auto-detect model name if not provided
    if args.model_name is None:
        if "en-indic" in str(args.model_dir).lower():
            args.model_name = "ai4bharat/indictrans2-en-indic-1B"
        elif "indic-en" in str(args.model_dir).lower():
            args.model_name = "ai4bharat/indictrans2-indic-en-1B"
        else:
            args.model_name = "ai4bharat/indictrans2-en-indic-1B"  # Default
    
    sample_sentences = [
        "My name is Ervin and I live in Bangalore.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you today?"
    ]
    sample_sentences = [
        "My name is ritik and I live in Bangalore.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you today?"
    ]
    
    logger.info(f"Initializing with device: {args.device}")
    logger.info(f"Using tokenizer: {args.model_name}")
    
    translator = IndicTrans2OpenVINO(
        model_dir=args.model_dir,
        device=args.device,
        model_name=args.model_name
    )
    
    if args.warmup > 0:
        translator.warmup(args.warmup)
    
    logger.info(f"Translating {len(sample_sentences)} sentences...")
    benchmark_data = translator.batch_translate(
        sample_sentences,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("Translation Results")
    print("="*60)
    
    for batch_idx, data in benchmark_data.items():
        print(f"\nBatch {batch_idx + 1}:")
        for src, tgt in zip(data["input_sentences"], data["translations"]):
            print(f"  SRC: {src}")
            print(f"  TGT: {tgt}")
            print()
        print(f"  Total Time: {data['times']['total']:.3f}s")
        print(f"  Input tokens: {data['tokens']['total_input_tokens']}")
        print(f"  Output tokens: {data['tokens']['total_output_tokens']}")