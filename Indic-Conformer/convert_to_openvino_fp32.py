#!/usr/bin/env python3
"""
Convert Indic Conformer ONNX models to OpenVINO IR format
FP32 VERSION - No compression for maximum accuracy
"""

import os
import sys
from pathlib import Path
import subprocess
import time

class ModelConverterFP32:
    def __init__(self, input_dir="assets", output_dir="openvino_models_fp32"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

    def convert_model(self, onnx_file, output_subdir, model_name):
        """Convert a single ONNX model to OpenVINO IR (FP32 - no compression)"""

        onnx_path = self.input_dir / onnx_file
        output_path = self.output_dir / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)

        xml_file = output_path / f"{model_name}.xml"
        bin_file = output_path / f"{model_name}.bin"

        # Check if already converted
        if xml_file.exists() and bin_file.exists():
            print(f"  SKIP - {model_name} (already exists)")
            self.stats['skipped'] += 1
            return True

        # Build conversion command using OpenVINO Model Converter (ovc)
        # NO FP16 compression - use FP32 for maximum accuracy
        cmd = [
            "ovc",
            str(onnx_path),
            "--output_model", str(output_path / model_name)
            # NO --compress_to_fp16 flag!
        ]

        try:
            print(f"  Converting {model_name} (FP32)...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                # Check output files exist
                if xml_file.exists() and bin_file.exists():
                    xml_size = xml_file.stat().st_size / (1024*1024)
                    bin_size = bin_file.stat().st_size / (1024*1024)
                    print(f"  OK   - {model_name} ({xml_size:.2f}MB + {bin_size:.2f}MB) [FP32]")
                    self.stats['success'] += 1
                    return True
                else:
                    print(f"  FAIL - {model_name} (output files not created)")
                    self.stats['failed'] += 1
                    return False
            else:
                print(f"  FAIL - {model_name}")
                print(f"       Error: {result.stderr[:200]}")
                self.stats['failed'] += 1
                return False

        except subprocess.TimeoutExpired:
            print(f"  FAIL - {model_name} (timeout)")
            self.stats['failed'] += 1
            return False
        except Exception as e:
            print(f"  FAIL - {model_name} ({e})")
            self.stats['failed'] += 1
            return False

    def convert_all(self):
        """Convert all models to FP32"""

        print("=" * 70)
        print("Converting ONNX Models to OpenVINO IR Format (FP32)")
        print("=" * 70)
        print()
        print("NOTE: This creates FP32 models for maximum accuracy")
        print("      Size: ~4.8 GB (vs 2.4 GB for FP16)")
        print("      Accuracy: Best on both CPU and GPU")
        print("      Speed: Slower GPU, same CPU performance")
        print()

        start_time = time.time()

        # Core models
        print("Phase 1: Core Models")
        print("-" * 70)

        self.stats['total'] += 1
        self.convert_model("encoder.onnx", "encoder", "encoder")

        self.stats['total'] += 1
        self.convert_model("ctc_decoder.onnx", "ctc_decoder", "ctc_decoder")

        self.stats['total'] += 1
        self.convert_model("rnnt_decoder.onnx", "rnnt_decoder", "rnnt_decoder")

        print()

        # RNNT components (not used by CTC, but convert for completeness)
        print("Phase 2: RNNT Components (optional)")
        print("-" * 70)

        self.stats['total'] += 1
        self.convert_model("joint_enc.onnx", "joint_enc", "joint_enc")

        self.stats['total'] += 1
        self.convert_model("joint_pre_net.onnx", "joint_pre_net", "joint_pre_net")

        self.stats['total'] += 1
        self.convert_model("joint_pred.onnx", "joint_pred", "joint_pred")

        print()

        # Language-specific joint_post_net models
        print("Phase 3: Language-Specific Models (22 languages)")
        print("-" * 70)

        languages = [
            'as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 'ks', 'mai',
            'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta',
            'te', 'ur'
        ]

        for lang in languages:
            self.stats['total'] += 1
            onnx_file = f"joint_post_net_{lang}.onnx"
            self.convert_model(
                onnx_file,
                f"joint_post_net_{lang}",
                f"joint_post_net_{lang}"
            )

        print()

        # Summary
        elapsed = time.time() - start_time

        print("=" * 70)
        print("Conversion Complete (FP32)")
        print("=" * 70)
        print()
        print(f"Total models:    {self.stats['total']}")
        print(f"Successful:      {self.stats['success']}")
        print(f"Failed:          {self.stats['failed']}")
        print(f"Skipped:         {self.stats['skipped']}")
        print(f"Time taken:      {elapsed:.1f} seconds")
        print()

        if self.stats['success'] == self.stats['total'] - self.stats['skipped']:
            print("All models converted successfully to FP32!")
            print()
            print("Benefits of FP32 models:")
            print("  - Maximum accuracy on both CPU and GPU")
            print("  - No precision loss")
            print("  - Same accuracy across all devices")
            print()
            print("Trade-offs:")
            print("  - 2x larger size (4.8 GB vs 2.4 GB)")
            print("  - Slower GPU inference (still faster than CPU)")
            print()
            print("To use FP32 models:")
            print("  model = IndicASROpenVINOOptimized(")
            print("      model_dir='openvino_models_fp32',")
            print("      device='GPU',")
            print("      use_native_preprocessing=True")
            print("  )")
        else:
            print(f"WARNING: {self.stats['failed']} models failed to convert")
            print("Check errors above for details")

        print()


if __name__ == '__main__':
    print()
    print("This script converts models to FP32 (no compression)")
    print("Use this for maximum accuracy on GPU")
    print()
    print("FP32 vs FP16:")
    print("  FP32: Better accuracy, larger size (4.8 GB), slower GPU")
    print("  FP16: Good accuracy, smaller size (2.4 GB), faster GPU")
    print()

    response = input("Continue with FP32 conversion? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        sys.exit(0)

    print()
    converter = ModelConverterFP32()
    converter.convert_all()
