"""
INT4 Quantization Script for Whisper Models using OpenVINO NNCF.

This script converts Whisper models to INT4 format for efficient edge deployment.
Supports various Whisper model sizes and optimization configurations.
"""

import os
import gc
import shutil
import argparse
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
from loguru import logger
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# OpenVINO imports - compatible with multiple versions
try:
    import openvino as ov
    from openvino import Core
    OV_VERSION = ov.__version__ if hasattr(ov, '__version__') else "unknown"
except ImportError:
    # Fallback for older OpenVINO versions
    from openvino.runtime import Core
    import openvino.runtime as ov
    OV_VERSION = getattr(ov, '__version__', "unknown (runtime)")

logger.info(f"OpenVINO version: {OV_VERSION}")

import nncf
from nncf import compress_weights
logger.info(f"NNCF version: {nncf.__version__}")

# Optional NNCF advanced features (may not be available in all versions)
try:
    from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
    from nncf.quantization.range_estimator import StatisticsType
    HAS_ADVANCED_NNCF = True
except ImportError:
    HAS_ADVANCED_NNCF = False
    AdvancedCompressionParameters = None
    StatisticsType = None


# Supported Whisper models
WHISPER_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base", 
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}

# Device-specific configurations for edge deployment
EDGE_CONFIGS = {
    "cpu_low_power": {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 128,
        "ratio": 1.0,
        "sensitivity_metric": nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION
    },
    "cpu_balanced": {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 64,
        "ratio": 0.8,
        "sensitivity_metric": nncf.SensitivityMetric.WEIGHT_QUANTIZATION_ERROR
    },
    "npu_optimized": {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 128,
        "ratio": 1.0,
        "all_layers": True
    },
    "gpu_hybrid": {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 32,
        "ratio": 0.9,
        "sensitivity_metric": nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE
    }
}


class WhisperQuantizer:
    """
    Handles INT4 quantization of Whisper models for OpenVINO deployment.
    """
    
    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        output_dir: str = "whisper_int4",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.device = device
        self.cache_dir = cache_dir
        
        self.processor = None
        self.model = None
        self.ov_model = None
        
    def load_model(self):
        """Load the Whisper model and processor."""
        logger.info(f"Loading model: {self.model_id}")
        
        self.processor = WhisperProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir
        )
        
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            cache_dir=self.cache_dir
        )
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
    def prepare_calibration_data(
        self,
        num_samples: int = 100,
        dataset_name: str = "librispeech_asr",
        dataset_config: str = "clean",
        split: str = "validation"
    ) -> List[dict]:
        """
        Prepare calibration dataset for quantization.
        
        Args:
            num_samples: Number of calibration samples
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split to use
            
        Returns:
            List of preprocessed input dictionaries
        """
        logger.info(f"Preparing calibration data from {dataset_name}")
        
        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=True
            )
        except Exception as e:
            logger.warning(f"Could not load {dataset_name}: {e}")
            logger.info("Generating synthetic calibration data")
            return self._generate_synthetic_data(num_samples)
        
        calibration_data = []
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            audio = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            
            # Preprocess audio
            inputs = self.processor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            
            calibration_data.append({
                "input_features": inputs.input_features,
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{num_samples} calibration samples")
        
        logger.info(f"Prepared {len(calibration_data)} calibration samples")
        return calibration_data
    
    def _generate_synthetic_data(self, num_samples: int) -> List[dict]:
        """Generate synthetic audio data for calibration when real data isn't available."""
        calibration_data = []
        
        for _ in range(num_samples):
            # Generate random audio-like data (white noise with some structure)
            duration_seconds = np.random.uniform(1, 10)
            num_samples_audio = int(duration_seconds * 16000)
            
            # Create structured noise that resembles speech patterns
            t = np.linspace(0, duration_seconds, num_samples_audio)
            audio = np.sin(2 * np.pi * 200 * t) * 0.3  # Base frequency
            audio += np.random.randn(num_samples_audio) * 0.1  # Add noise
            audio = audio.astype(np.float32)
            
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            calibration_data.append({
                "input_features": inputs.input_features,
            })
        
        return calibration_data
    
    def export_to_openvino(self) -> ov.Model:
        """Export PyTorch model to OpenVINO IR format."""
        logger.info("Exporting model to OpenVINO format...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 80, 3000)  # Whisper mel spectrogram input
        
        # Export encoder
        encoder_path = self.output_dir / "encoder_fp32.xml"
        
        with torch.no_grad():
            self.model.eval()
            
            ov_model = ov.convert_model(
                self.model,
                example_input={"input_features": dummy_input},
                input=[("input_features", ov.PartialShape([1, 80, -1]))]
            )
        
        logger.info("Model exported to OpenVINO format")
        return ov_model
    
    def quantize_int4(
        self,
        calibration_data: List[dict],
        config_name: str = "cpu_balanced"
    ) -> ov.Model:
        """
        Quantize the model to INT4 using NNCF.
        
        Args:
            calibration_data: List of calibration samples
            config_name: Quantization configuration preset
            
        Returns:
            Quantized OpenVINO model
        """
        logger.info(f"Quantizing model to INT4 using config: {config_name}")
        
        config = EDGE_CONFIGS.get(config_name, EDGE_CONFIGS["cpu_balanced"])
        
        # First export to OpenVINO
        ov_model = self.export_to_openvino()
        
        # Create calibration dataset for NNCF
        def transform_fn(data_item):
            return {"input_features": data_item["input_features"].numpy()}
        
        calibration_dataset = nncf.Dataset(calibration_data, transform_fn)
        
        # Apply INT4 weight compression
        logger.info("Applying INT4 weight compression...")
        
        compressed_model = compress_weights(
            ov_model,
            mode=config["mode"],
            group_size=config.get("group_size", 128),
            ratio=config.get("ratio", 1.0),
            dataset=calibration_dataset,
            sensitivity_metric=config.get("sensitivity_metric"),
            all_layers=config.get("all_layers", False)
        )
        
        logger.info("INT4 quantization complete")
        return compressed_model
    
    def quantize_full_pipeline(
        self,
        calibration_data: List[dict],
        config_name: str = "cpu_balanced"
    ):
        """
        Quantize both encoder and decoder components.
        
        For Whisper, we need to handle encoder and decoder separately
        for optimal performance.
        """
        logger.info("Quantizing full Whisper pipeline...")
        
        config = EDGE_CONFIGS.get(config_name, EDGE_CONFIGS["cpu_balanced"])
        
        # Quantize encoder
        logger.info("Quantizing encoder...")
        encoder = self.model.get_encoder()
        
        dummy_input = torch.randn(1, 80, 3000)
        with torch.no_grad():
            encoder_ov = ov.convert_model(
                encoder,
                example_input=dummy_input,
                input=[ov.PartialShape([1, 80, -1])]
            )
        
        def encoder_transform(data_item):
            return data_item["input_features"].numpy()
        
        encoder_dataset = nncf.Dataset(calibration_data, encoder_transform)
        
        encoder_int4 = compress_weights(
            encoder_ov,
            mode=config["mode"],
            group_size=config.get("group_size", 128),
            ratio=config.get("ratio", 1.0),
            dataset=encoder_dataset,
            all_layers=config.get("all_layers", False)
        )
        
        # Save encoder
        encoder_path = self.output_dir / "encoder"
        encoder_path.mkdir(parents=True, exist_ok=True)
        ov.save_model(encoder_int4, encoder_path / "openvino_encoder_model.xml")
        logger.info(f"Encoder saved to {encoder_path}")
        
        # Quantize decoder
        logger.info("Quantizing decoder...")
        decoder = self.model.get_decoder()
        
        # Decoder requires encoder hidden states as input
        seq_len = 512
        hidden_size = self.model.config.d_model
        
        dummy_decoder_input = torch.tensor([[1, 2, 3]])  # Decoder input IDs
        dummy_encoder_hidden = torch.randn(1, seq_len, hidden_size)
        
        with torch.no_grad():
            decoder_ov = ov.convert_model(
                decoder,
                example_input={
                    "input_ids": dummy_decoder_input,
                    "encoder_hidden_states": dummy_encoder_hidden
                }
            )
        
        decoder_int4 = compress_weights(
            decoder_ov,
            mode=config["mode"],
            group_size=config.get("group_size", 128),
            ratio=config.get("ratio", 1.0),
            all_layers=config.get("all_layers", False)
        )
        
        # Save decoder
        decoder_path = self.output_dir / "decoder"
        decoder_path.mkdir(parents=True, exist_ok=True)
        ov.save_model(decoder_int4, decoder_path / "openvino_decoder_model.xml")
        logger.info(f"Decoder saved to {decoder_path}")
        
        # Save processor/tokenizer config
        self.processor.save_pretrained(self.output_dir)
        logger.info("Processor config saved")
        
        # Save generation config
        self.model.generation_config.save_pretrained(self.output_dir)
        logger.info("Generation config saved")
        
        logger.info(f"Full pipeline quantized and saved to {self.output_dir}")
    
    def save_model(self, model: ov.Model, name: str = "model"):
        """Save quantized model to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = self.output_dir / f"{name}.xml"
        ov.save_model(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save processor
        self.processor.save_pretrained(self.output_dir)
        logger.info("Processor saved")
        
    def convert_with_optimum(self):
        """
        Alternative conversion using Optimum Intel library.
        This is recommended for seamless integration with OpenVINO GenAI.
        """
        try:
            from optimum.intel import OVWeightQuantizationConfig
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq
            
            logger.info("Using Optimum Intel for conversion with INT4 quantization...")
            
            # Configure INT4 quantization
            quantization_config = OVWeightQuantizationConfig(
                bits=4,
                sym=False,
                group_size=128,
                ratio=1.0
            )
            
            # Export and quantize in one step
            ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                export=True,
                quantization_config=quantization_config,
                cache_dir=self.cache_dir
            )
            
            # Save the model
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ov_model.save_pretrained(self.output_dir)
            self.processor.save_pretrained(self.output_dir)
            
            logger.info(f"Model quantized and saved to {self.output_dir}")
            
        except ImportError:
            logger.warning("Optimum Intel not available, using NNCF directly")
            raise
    
    def benchmark(self, num_iterations: int = 10):
        """Benchmark the quantized model."""
        logger.info("Running benchmark...")
        
        core = Core()
        model_path = self.output_dir / "encoder" / "openvino_encoder_model.xml"
        
        if not model_path.exists():
            model_path = self.output_dir / "model.xml"
            
        if not model_path.exists():
            logger.error("No model found for benchmarking")
            return
            
        compiled_model = core.compile_model(model_path, self.device.upper())
        
        # Create dummy input
        dummy_input = np.random.randn(1, 80, 3000).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            compiled_model(dummy_input)
        
        # Benchmark
        import time
        times = []
        
        for i in range(num_iterations):
            start = time.perf_counter()
            compiled_model(dummy_input)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        logger.info(f"Benchmark Results ({self.device.upper()}):")
        logger.info(f"  Average latency: {avg_time:.2f} ms")
        logger.info(f"  Std deviation: {std_time:.2f} ms")
        logger.info(f"  Throughput: {1000/avg_time:.2f} inferences/sec")
        
        # Model size comparison
        fp32_size = sum(f.stat().st_size for f in self.output_dir.rglob("*.bin"))
        logger.info(f"  Model size: {fp32_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Whisper models to INT4 for OpenVINO edge deployment"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="small",
        choices=list(WHISPER_MODELS.keys()) + ["custom"],
        help="Whisper model size or 'custom' for custom model ID"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Custom HuggingFace model ID (use with --model custom)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="whisper_int4",
        help="Output directory for quantized model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="cpu_balanced",
        choices=list(EDGE_CONFIGS.keys()),
        help="Quantization configuration preset"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "npu"],
        help="Target device for benchmarking"
    )
    
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=50,
        help="Number of samples for calibration"
    )
    
    parser.add_argument(
        "--use-optimum",
        action="store_true",
        help="Use Optimum Intel for conversion (recommended)"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after quantization"
    )
    
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Quantize both encoder and decoder separately"
    )
    
    args = parser.parse_args()
    
    # Determine model ID
    if args.model == "custom":
        if not args.model_id:
            raise ValueError("--model-id required when using --model custom")
        model_id = args.model_id
    else:
        model_id = WHISPER_MODELS[args.model]
    
    logger.info(f"Starting INT4 quantization for {model_id}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize quantizer
    quantizer = WhisperQuantizer(
        model_id=model_id,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Load model
    quantizer.load_model()
    
    if args.use_optimum:
        # Use Optimum Intel (simpler, recommended)
        try:
            quantizer.convert_with_optimum()
        except ImportError:
            logger.info("Falling back to NNCF quantization")
            calibration_data = quantizer.prepare_calibration_data(
                num_samples=args.num_calibration_samples
            )
            if args.full_pipeline:
                quantizer.quantize_full_pipeline(calibration_data, args.config)
            else:
                quantized_model = quantizer.quantize_int4(calibration_data, args.config)
                quantizer.save_model(quantized_model)
    else:
        # Use NNCF directly
        calibration_data = quantizer.prepare_calibration_data(
            num_samples=args.num_calibration_samples
        )
        
        if args.full_pipeline:
            quantizer.quantize_full_pipeline(calibration_data, args.config)
        else:
            quantized_model = quantizer.quantize_int4(calibration_data, args.config)
            quantizer.save_model(quantized_model)
    
    # Cleanup
    gc.collect()
    
    # Run benchmark if requested
    if args.benchmark:
        quantizer.benchmark()
    
    logger.info("Quantization complete!")
    logger.info(f"Quantized model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
