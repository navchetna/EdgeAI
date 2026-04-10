"""
OpenVINO inference class for IndicTrans2 models - Synchronous GPU-Optimized Batching.

This module provides an OpenVINO-based inference engine with efficient synchronous
batching for GPU acceleration. Processes entire batches together as single operations
rather than per-sequence async requests.

Usage:
    from indictrans2_openvino_sync import IndicTrans2OpenVINO
    
    translator = IndicTrans2OpenVINO(
        model_dir="./openvino_models/indictrans2-indic-indic-1B-fp16/optimum",
        device="GPU"
    )
    
    translations = translator.translate(
        ["जब मैं छोटा था, तो मैं हर दिन पार्क में जाता था।"],
        src_lang="hin_Deva",
        tgt_lang="ben_Beng"
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
    OpenVINO-based inference engine for IndicTrans2 with synchronous GPU batching.
    
    Args:
        model_dir: Path to directory containing converted OpenVINO models
        device: Target device for inference ("CPU", "GPU", "AUTO")
        max_length: Maximum sequence length for generation
        num_beams: Number of beams for beam search
    """
    
    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "GPU",
        max_length: int = 128,
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
            
            source = model_name if model_name else "ai4bharat/indictrans2-indic-indic-1B"
            
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
        
        encoder_path = self.model_dir / "encoder.xml"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
        
        logger.info(f"Loading encoder from: {encoder_path}")
        self.encoder = core.compile_model(core.read_model(encoder_path), self.device)
        logger.info("  ✓ Encoder compiled")
        
        decoder_prefill_path = self.model_dir / "decoder_prefill.xml"
        if not decoder_prefill_path.exists():
            raise FileNotFoundError(f"Decoder prefill model not found: {decoder_prefill_path}")
        
        logger.info(f"Loading decoder prefill from: {decoder_prefill_path}")
        self.decoder_prefill = core.compile_model(core.read_model(decoder_prefill_path), self.device)
        logger.info("  ✓ Decoder prefill compiled")
        
        decoder_decode_path = self.model_dir / "decoder_decode.xml"
        if not decoder_decode_path.exists():
            raise FileNotFoundError(f"Decoder decode model not found: {decoder_decode_path}")
        
        logger.info(f"Loading decoder decode from: {decoder_decode_path}")
        self.decoder_decode = core.compile_model(core.read_model(decoder_decode_path), self.device)
        logger.info("  ✓ Decoder decode compiled")
        
        model_config = self.config.get("model_config", {})
        self.num_layers = model_config.get("decoder_layers", 18)
        self.num_heads = model_config.get("decoder_attention_heads", 16)
        self.head_dim = model_config.get("encoder_embed_dim", 1024) // self.num_heads
        self.vocab_size = model_config.get("decoder_vocab_size", 256000)
        self.pad_token_id = model_config.get("pad_token_id", 1)
        self.eos_token_id = model_config.get("eos_token_id", 2)
        self.bos_token_id = model_config.get("bos_token_id", 0)
        self.decoder_start_token_id = model_config.get("decoder_start_token_id", 2)
    
    def preprocess(
        self,
        sentences: List[str],
        src_lang: str,
        tgt_lang: str
    ) -> Dict[str, np.ndarray]:
        """Preprocess input sentences for translation."""
        preprocessed = self.ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        
        inputs = self.tokenizer(
            preprocessed,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    
    def postprocess(
        self,
        token_ids: np.ndarray,
        tgt_lang: str
    ) -> List[str]:
        """Postprocess generated token IDs to text."""
        decoded = self.tokenizer.batch_decode(
            token_ids.tolist(),
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
        """Run decoder prefill step (first token generation)."""
        results = self.decoder_prefill([
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask
        ])
        
        logits = results[0]
        past_key_values = []
        
        for i in range(self.num_layers):
            idx = 1 + i * 4
            past_key_values.append((
                results[idx],
                results[idx + 1],
                results[idx + 2],
                results[idx + 3]
            ))
        
        return logits, past_key_values
    
    def decode_step(
        self,
        decoder_input_ids: np.ndarray,
        encoder_hidden_states: np.ndarray,
        encoder_attention_mask: np.ndarray,
        past_key_values: List[Tuple[np.ndarray, ...]]
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, ...]]]:
        """Run single decoder step with KV-cache for entire batch."""
        inputs = [
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask,
        ]
        
        for self_k, self_v, cross_k, cross_v in past_key_values:
            inputs.extend([self_k, self_v, cross_k, cross_v])
        
        results = self.decoder_decode(inputs)
        logits = results[0]
        
        new_past_key_values = []
        for i in range(self.num_layers):
            idx = 1 + i * 4
            new_past_key_values.append((
                results[idx],
                results[idx + 1],
                results[idx + 2],
                results[idx + 3]
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
        Generate translations using synchronous batched decoding.
        Processes entire batch together as single matrix operations.
        """
        max_length = max_length or self.max_length
        num_beams = num_beams or self.num_beams
        
        batch_size = input_ids.shape[0]
        
        encoder_hidden_states = self.encode(input_ids, attention_mask)
        
        decoder_input_ids = np.full((batch_size, 1), self.decoder_start_token_id, dtype=np.int64)
        
        logits, past_key_values = self.decode_prefill(
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask
        )
        
        next_token_logits = logits[:, -1, :]
        next_tokens = np.argmax(next_token_logits, axis=-1, keepdims=True)
        
        finished = (next_tokens[:, 0] == self.eos_token_id)
        active_mask = ~finished
        
        generated_tokens = [next_tokens]
        
        for step in range(max_length - 1):
            if not np.any(active_mask):
                break
            
            logits, past_key_values = self.decode_step(
                next_tokens,
                encoder_hidden_states,
                attention_mask,
                past_key_values
            )
            
            next_token_logits = logits[:, -1, :]
            next_tokens = np.argmax(next_token_logits, axis=-1, keepdims=True)
            
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
        Synchronous processing: one batch at a time, full batch operations.
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
            output_tokens = (output_ids != self.pad_token_id).sum(axis=1).tolist()
            
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
        dummy_sentences = ["यह एक परीक्षण वाक्य है।"] * 2
        
        for _ in range(num_iterations):
            _ = self.translate(dummy_sentences, "hin_Deva", "ben_Beng")
        
        logger.info("Warmup complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IndicTrans2 OpenVINO Synchronous Inference")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to converted OpenVINO models")
    parser.add_argument("--device", type=str, default="GPU", choices=["CPU", "GPU", "AUTO"])
    parser.add_argument("--src-lang", type=str, default="hin_Deva")
    parser.add_argument("--tgt-lang", type=str, default="ben_Beng")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model-name", type=str, default="ai4bharat/indictrans2-indic-indic-1B")
    
    args = parser.parse_args()
    
    sample_hi_sents = [
        "जब मैं छोटा था, तो मैं हर दिन पार्क में जाता था।",
        "उसके पास कई पुरानी किताबें हैं, जो उसने अपने पूर्वजों से विरासत में पाई हैं।",
        "मैं समझ नहीं पा रहा हूँ कि अपनी समस्या कैसे हल करूँ।",
        "वह बहुत मेहनती और बुद्धिमान है, इसी कारण उसे सभी अच्छे अंक मिले हैं।",
        "हमने पिछले हफ्ते एक नई फिल्म देखी, जो बहुत प्रेरणादायक थी।",
        "अगर तुम उस समय मुझसे मिले होते, तो हम बाहर खाने चले जाते।",
        "वह अपनी बहन के साथ एक नई साड़ी खरीदने बाजार गई।",
    ]
    
    logger.info(f"Initializing with device: {args.device}")
    translator = IndicTrans2OpenVINO(
        model_dir=args.model_dir,
        device=args.device,
        model_name=args.model_name
    )
    
    if args.warmup > 0:
        translator.warmup(args.warmup)
    
    logger.info(f"Translating {len(sample_hi_sents)} sentences...")
    benchmark_data = translator.batch_translate(
        sample_hi_sents,
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