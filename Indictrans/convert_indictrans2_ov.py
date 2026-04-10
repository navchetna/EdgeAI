"""
Script to convert IndicTrans2 models to OpenVINO format.

This script downloads the pretrained IndicTrans2 models and converts them to OpenVINO IR format
for optimized inference on Intel hardware (CPU and GPU).

Usage:
    python convert_indictrans2_ov.py --output-dir ./openvino_models/indictrans2-indic-indic-1B-fp16
    python convert_indictrans2_ov.py --model-name ai4bharat/indictrans2-en-indic-1B --output-dir ./openvino_models/indictrans2-en-indic-1B-fp32 --precision FP32
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import openvino
        logger.info(f"OpenVINO version: {openvino.__version__}")
    except ImportError:
        missing.append("openvino")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        from IndicTransToolkit.processor import IndicProcessor
        logger.info("IndicTransToolkit is available")
    except ImportError:
        missing.append("IndicTransToolkit")
    
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def download_model(model_name: str, output_dir: Path) -> Path:
    """Download pretrained model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    
    logger.info(f"Downloading model from {model_name}...")
    
    model_dir = output_dir / "pytorch_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = snapshot_download(
        repo_id=model_name,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False
    )
    
    logger.info(f"Model downloaded to: {local_path}")
    return Path(local_path)


def convert_encoder(model, output_dir: Path, precision: str = "FP16"):
    """
    Convert the encoder part of IndicTrans2 to OpenVINO.
    
    The encoder processes input tokens and produces encoder_hidden_states
    that are used by the decoder via cross-attention.
    
    Input: input_ids, attention_mask
    Output: encoder_hidden_states
    """
    import torch
    import torch.nn as nn
    from openvino import convert_model, save_model
    
    logger.info("Converting Encoder...")
    
    # Get config values - IndicTrans uses encoder_embed_dim, not d_model
    config = model.config
    encoder_embed_dim = config.encoder_embed_dim
    
    class EncoderWrapper(nn.Module):
        """
        Wrapper for IndicTrans2 encoder.
        
        Input shapes:
            - input_ids: (batch, seq_len)
            - attention_mask: (batch, seq_len)
        
        Output shape:
            - encoder_hidden_states: (batch, seq_len, encoder_embed_dim)
        """
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        
        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            # The encoder handles embedding internally
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            return encoder_outputs.last_hidden_state
    
    encoder_wrapper = EncoderWrapper(model.model.encoder)
    encoder_wrapper.eval()
    
    # Example inputs with dynamic shapes
    batch_size = 1
    seq_len = 128
    
    example_input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    example_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Convert with dynamic shapes
    ov_encoder = convert_model(
        encoder_wrapper,
        example_input=(example_input_ids, example_attention_mask),
        input=[
            (-1, -1),  # input_ids: dynamic batch and seq_len
            (-1, -1),  # attention_mask: dynamic batch and seq_len
        ]
    )
    
    # Save model
    encoder_path = output_dir / "encoder.xml"
    save_model(ov_encoder, encoder_path)
    logger.info(f"  ✓ Encoder saved to: {encoder_path}")
    
    return encoder_path


def convert_decoder(model, output_dir: Path, precision: str = "FP16"):
    """
    Convert the decoder part of IndicTrans2 to OpenVINO with KV-cache support.
    
    The decoder generates tokens autoregressively using:
    1. Previous decoder hidden states (self-attention with KV-cache)
    2. Encoder hidden states (cross-attention)
    
    We export two variants:
    - decoder_prefill: First step without KV-cache (or with empty cache)
    - decoder_decode: Subsequent steps with KV-cache
    """
    import torch
    import torch.nn as nn
    from openvino import convert_model, save_model
    
    logger.info("Converting Decoder...")
    
    config = model.config
    num_layers = config.decoder_layers
    num_heads = config.decoder_attention_heads
    encoder_embed_dim = config.encoder_embed_dim  # IndicTrans uses encoder_embed_dim for both
    vocab_size = config.decoder_vocab_size
    
    logger.info(f"  Decoder config: layers={num_layers}, heads={num_heads}, "
                f"embed_dim={encoder_embed_dim}")
    
    # ========================================================================
    # 1. Decoder Prefill (first step, no KV-cache or empty cache)
    # ========================================================================
    logger.info("  Exporting decoder prefill model...")
    
    class DecoderPrefillWrapper(nn.Module):
        """
        Decoder for the first token generation (prefill phase).
        
        Inputs:
            - decoder_input_ids: (batch, seq_len) - usually just BOS token or initial prompt
            - encoder_hidden_states: (batch, enc_seq_len, encoder_embed_dim)
            - encoder_attention_mask: (batch, enc_seq_len)
        
        Outputs:
            - logits: (batch, seq_len, vocab_size)
            - past_key_values: KV-cache for each layer
        """
        def __init__(self, decoder, lm_head):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.num_layers = len(decoder.layers)
        
        def forward(
            self,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor
        ):
            # Decode with no past_key_values (prefill)
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=None,
                use_cache=True,
                return_dict=True
            )
            
            # Project to vocabulary
            logits = self.lm_head(decoder_outputs.last_hidden_state)
            
            # Flatten KV-cache for output
            # IndicTrans decoder returns 4 tensors per layer: (self_attn_k, self_attn_v, cross_attn_k, cross_attn_v)
            outputs = [logits]
            for layer_past in decoder_outputs.past_key_values:
                for tensor in layer_past:
                    outputs.append(tensor)
            
            return tuple(outputs)
    
    prefill_wrapper = DecoderPrefillWrapper(
        model.model.decoder,
        model.lm_head
    )
    prefill_wrapper.eval()
    
    # Example inputs
    batch_size = 1
    dec_seq_len = 1  # Usually just the BOS token for first step
    enc_seq_len = 128
    
    example_decoder_input_ids = torch.ones(batch_size, dec_seq_len, dtype=torch.long)
    example_encoder_hidden = torch.randn(batch_size, enc_seq_len, encoder_embed_dim)
    example_encoder_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.long)
    
    ov_decoder_prefill = convert_model(
        prefill_wrapper,
        example_input=(example_decoder_input_ids, example_encoder_hidden, example_encoder_mask),
        input=[
            (-1, -1),           # decoder_input_ids: dynamic batch and seq_len
            (-1, -1, encoder_embed_dim),  # encoder_hidden_states: dynamic batch and enc_seq_len
            (-1, -1),           # encoder_attention_mask: dynamic batch and enc_seq_len
        ]
    )
    
    prefill_path = output_dir / "decoder_prefill.xml"
    save_model(ov_decoder_prefill, prefill_path)
    logger.info(f"    ✓ Decoder prefill saved to: {prefill_path}")
    
    # ========================================================================
    # 2. Decoder Decode (subsequent steps with KV-cache)
    # ========================================================================
    logger.info("  Exporting decoder decode model (with KV-cache)...")
    
    class DecoderDecodeWrapper(nn.Module):
        """
        Decoder for subsequent token generation (decode phase with KV-cache).
        
        Inputs:
            - decoder_input_ids: (batch, 1) - single token
            - encoder_hidden_states: (batch, enc_seq_len, encoder_embed_dim)
            - encoder_attention_mask: (batch, enc_seq_len)
            - past_key_values: KV-cache tensors for each layer (4 per layer)
        
        Outputs:
            - logits: (batch, 1, vocab_size)
            - updated past_key_values
        """
        def __init__(self, decoder, lm_head, num_layers):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.num_layers = num_layers
        
        def forward(
            self,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
            *flat_kv_cache
        ):
            # Reconstruct past_key_values from flat tensors
            # Each layer has 4 tensors: self_attn_k, self_attn_v, cross_attn_k, cross_attn_v
            past_key_values = []
            for i in range(self.num_layers):
                idx = i * 4
                layer_cache = (
                    flat_kv_cache[idx],      # self_attn_k
                    flat_kv_cache[idx + 1],  # self_attn_v
                    flat_kv_cache[idx + 2],  # cross_attn_k
                    flat_kv_cache[idx + 3]   # cross_attn_v
                )
                past_key_values.append(layer_cache)
            
            # Decode with cache
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=tuple(past_key_values),
                use_cache=True,
                return_dict=True
            )
            
            # Project to vocabulary
            logits = self.lm_head(decoder_outputs.last_hidden_state)
            
            # Flatten updated KV-cache for output
            outputs = [logits]
            for layer_past in decoder_outputs.past_key_values:
                for tensor in layer_past:
                    outputs.append(tensor)
            
            return tuple(outputs)
    
    decode_wrapper = DecoderDecodeWrapper(
        model.model.decoder,
        model.lm_head,
        num_layers
    )
    decode_wrapper.eval()
    
    # Example inputs for decode step
    cache_len = 128  # Length of cached keys/values
    head_dim = encoder_embed_dim // num_heads
    
    example_decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long)
    example_encoder_hidden = torch.randn(batch_size, enc_seq_len, encoder_embed_dim)
    example_encoder_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.long)
    
    # Build example KV-cache inputs (4 tensors per layer)
    example_kv_inputs = [
        example_decoder_input_ids,
        example_encoder_hidden,
        example_encoder_mask
    ]
    
    # Add KV tensors for each layer
    for _ in range(num_layers):
        # Self-attention key and value
        example_kv_inputs.append(torch.randn(batch_size, num_heads, cache_len, head_dim))  # self_attn_k
        example_kv_inputs.append(torch.randn(batch_size, num_heads, cache_len, head_dim))  # self_attn_v
        # Cross-attention key and value (these don't grow with generation)
        example_kv_inputs.append(torch.randn(batch_size, num_heads, enc_seq_len, head_dim))  # cross_attn_k
        example_kv_inputs.append(torch.randn(batch_size, num_heads, enc_seq_len, head_dim))  # cross_attn_v
    
    # Build input shapes
    input_shapes = [
        (-1, -1),           # decoder_input_ids: dynamic batch and seq_len (usually 1)
        (-1, -1, encoder_embed_dim),  # encoder_hidden_states
        (-1, -1),           # encoder_attention_mask
    ]
    
    # Add KV-cache shapes (dynamic batch and cache_len)
    for i in range(num_layers):
        # Self-attention KV (grows with generation)
        input_shapes.append((-1, num_heads, -1, head_dim))  # self_attn_k: dynamic cache_len
        input_shapes.append((-1, num_heads, -1, head_dim))  # self_attn_v: dynamic cache_len
        # Cross-attention KV (fixed to encoder seq len)
        input_shapes.append((-1, num_heads, -1, head_dim))  # cross_attn_k: fixed to enc_seq_len
        input_shapes.append((-1, num_heads, -1, head_dim))  # cross_attn_v: fixed to enc_seq_len
    
    ov_decoder_decode = convert_model(
        decode_wrapper,
        example_input=tuple(example_kv_inputs),
        input=input_shapes
    )
    
    decode_path = output_dir / "decoder_decode.xml"
    save_model(ov_decoder_decode, decode_path)
    logger.info(f"    ✓ Decoder decode saved to: {decode_path}")
    
    return {
        "prefill": prefill_path,
        "decode": decode_path
    }


def convert_full_model(model, output_dir: Path, precision: str = "FP16"):
    """
    Convert the complete model (encoder + decoder) for simple use cases.
    
    This exports the model for a single forward pass without generate() loop.
    Useful for simple translation with fixed max_length.
    """
    import torch
    import torch.nn as nn
    from openvino import convert_model, save_model
    
    logger.info("Converting full model (encoder + decoder for single forward pass)...")
    
    class FullModelWrapper(nn.Module):
        """
        Wrapper for the complete seq2seq model.
        
        Inputs:
            - input_ids: (batch, seq_len)
            - attention_mask: (batch, seq_len)
            - decoder_input_ids: (batch, dec_seq_len)
        
        Outputs:
            - logits: (batch, dec_seq_len, vocab_size)
        """
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_input_ids: torch.Tensor
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            return outputs.logits
    
    full_wrapper = FullModelWrapper(model)
    full_wrapper.eval()
    
    # Example inputs
    batch_size = 1
    seq_len = 128
    dec_seq_len = 128
    
    example_input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    example_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    example_decoder_input_ids = torch.ones(batch_size, dec_seq_len, dtype=torch.long)
    
    ov_full = convert_model(
        full_wrapper,
        example_input=(example_input_ids, example_attention_mask, example_decoder_input_ids),
        input=[
            (-1, -1),  # input_ids
            (-1, -1),  # attention_mask
            (-1, -1),  # decoder_input_ids
        ]
    )
    
    full_path = output_dir / "full_model.xml"
    save_model(ov_full, full_path)
    logger.info(f"  ✓ Full model saved to: {full_path}")
    
    return full_path


def create_config_file(
    output_dir: Path,
    encoder_path: Optional[Path],
    decoder_paths: Optional[Dict[str, Path]],
    full_model_path: Optional[Path],
    model_config,
    precision: str
):
    """Create a configuration file for the converted models."""
    import json
    
    config = {
        "precision": precision,
        "model_type": "indictrans2",
        "architecture": "encoder-decoder",
        "models": {
            "encoder": str(encoder_path) if encoder_path else None,
            "decoder_prefill": str(decoder_paths["prefill"]) if decoder_paths else None,
            "decoder_decode": str(decoder_paths["decode"]) if decoder_paths else None,
            "full_model": str(full_model_path) if full_model_path else None,
        },
        "model_config": {
            "encoder_vocab_size": model_config.encoder_vocab_size,
            "decoder_vocab_size": model_config.decoder_vocab_size,
            "encoder_embed_dim": model_config.encoder_embed_dim,
            "decoder_embed_dim": model_config.decoder_embed_dim,
            "encoder_layers": model_config.encoder_layers,
            "decoder_layers": model_config.decoder_layers,
            "encoder_attention_heads": model_config.encoder_attention_heads,
            "decoder_attention_heads": model_config.decoder_attention_heads,
            "encoder_ffn_dim": model_config.encoder_ffn_dim,
            "decoder_ffn_dim": model_config.decoder_ffn_dim,
            "max_source_positions": model_config.max_source_positions,
            "max_target_positions": model_config.max_target_positions,
            "pad_token_id": model_config.pad_token_id,
            "eos_token_id": model_config.eos_token_id,
            "bos_token_id": model_config.bos_token_id,
            "decoder_start_token_id": model_config.decoder_start_token_id,
        },
        "inference": {
            "use_cache": True,
            "dynamic_shapes": True,
            "supports_kv_cache": True,
        }
    }
    
    config_path = output_dir / "openvino_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to: {config_path}")
    return config_path


def convert_all_optimum(model_name: str, output_dir: Path, precision: str = "FP16"):
    """
    Convert all IndicTrans2 components using optimum-intel style export.
    
    This is the recommended approach for Intel GPU inference with dynamic shapes.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    logger.info("="*60)
    logger.info("Converting IndicTrans2 with optimum-intel style export")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use FP32 for conversion, quantize later if needed
    )
    model.eval()
    
    optimum_dir = output_dir / "optimum"
    optimum_dir.mkdir(parents=True, exist_ok=True)
    
    converted = {}
    
    # Convert encoder
    try:
        encoder_path = convert_encoder(model, optimum_dir, precision)
        converted["encoder"] = encoder_path
    except Exception as e:
        logger.error(f"Failed to convert encoder: {e}")
        import traceback
        traceback.print_exc()
        converted["encoder"] = None
    
    # Convert decoder
    try:
        decoder_paths = convert_decoder(model, optimum_dir, precision)
        converted["decoder"] = decoder_paths
    except Exception as e:
        logger.error(f"Failed to convert decoder: {e}")
        import traceback
        traceback.print_exc()
        converted["decoder"] = None
    
    # Convert full model (optional, for simple use cases)
    try:
        full_path = convert_full_model(model, optimum_dir, precision)
        converted["full_model"] = full_path
    except Exception as e:
        logger.warning(f"Failed to convert full model: {e}")
        converted["full_model"] = None
    
    # Create config
    create_config_file(
        optimum_dir,
        converted.get("encoder"),
        converted.get("decoder"),
        converted.get("full_model"),
        model.config,
        precision
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Conversion Summary")
    logger.info("="*60)
    
    successful = []
    if converted.get("encoder"):
        successful.append("encoder")
    if converted.get("decoder"):
        successful.append("decoder (prefill + decode)")
    if converted.get("full_model"):
        successful.append("full_model")
    
    if successful:
        logger.info(f"✓ Successfully converted: {', '.join(successful)}")
    else:
        logger.warning("✗ No models were successfully converted")
    
    logger.info(f"\nAll models saved to: {optimum_dir}")
    
    return optimum_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert IndicTrans2 models to OpenVINO format"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ai4bharat/indictrans2-indic-indic-1B",
        help="HuggingFace model name (e.g., ai4bharat/indictrans2-indic-indic-1B)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./openvino_models",
        help="Directory to save converted models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["GPU", "CPU", "AUTO"],
        help="Target device for optimization"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        choices=["FP32", "FP16"],
        help="Model precision"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading models (use existing)"
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=["encoder", "decoder", "full_model"],
        help="Components to convert (encoder, decoder, full_model)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target device: {args.device}")
    logger.info(f"Precision: {args.precision}")
    
    # Convert all components
    optimum_dir = convert_all_optimum(args.model_name, output_dir, args.precision)
    
    logger.info("\n" + "="*60)
    logger.info("Conversion Complete!")
    logger.info("="*60)
    logger.info(f"\nTo use the converted models:")
    logger.info(f"  from IndicTransToolkit import IndicProcessor")
    logger.info(f"  from transformers import AutoTokenizer")
    logger.info(f"  import openvino as ov")
    logger.info(f"")
    logger.info(f"  # Load OpenVINO models")
    logger.info(f"  encoder = ov.Core().read_model('{optimum_dir}/encoder.xml')")
    logger.info(f"  decoder_prefill = ov.Core().read_model('{optimum_dir}/decoder_prefill.xml')")
    logger.info(f"  decoder_decode = ov.Core().read_model('{optimum_dir}/decoder_decode.xml')")
    logger.info(f"")
    logger.info(f"  # Or use the provided IndicTrans2OpenVINO inference class")
    logger.info(f"  from indictrans2_openvino import IndicTrans2OpenVINO")
    logger.info(f"  translator = IndicTrans2OpenVINO('{optimum_dir}')")


if __name__ == "__main__":
    main()