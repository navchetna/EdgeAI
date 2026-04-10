#!/usr/bin/env python3
"""
Convert Indic Conformer ONNX models to OpenVINO IR format
"""

import os
import sys
from pathlib import Path
import subprocess
import time

class ModelConverter:
    def __init__(self, input_dir="assets", output_dir="openvino_models"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

    def convert_model(self, onnx_file, output_subdir, model_name, use_fp16=True):
        """Convert a single ONNX model to OpenVINO IR"""

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
        cmd = [
            "ovc",
            str(onnx_path),
            "--output_model", str(output_path / model_name)
        ]

        if use_fp16:
            cmd.extend(["--compress_to_fp16"])

        try:
            print(f"  Converting {model_name}...")
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
                    print(f"  OK   - {model_name} ({xml_size:.2f}MB + {bin_size:.2f}MB)")
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
        """Convert all models"""

        print("=" * 70)
        print("Converting ONNX Models to OpenVINO IR Format")
        print("=" * 70)
        print()

        start_time = time.time()

        # Core models
        print("Phase 1: Core Models")
        print("-" * 70)

        core_models = [
            ("encoder.onnx", "encoder", "encoder"),
            ("ctc_decoder.onnx", "ctc_decoder", "ctc_decoder"),
            ("rnnt_decoder.onnx", "rnnt_decoder", "rnnt_decoder"),
            ("joint_enc.onnx", "joint_enc", "joint_enc"),
            ("joint_pred.onnx", "joint_pred", "joint_pred"),
            ("joint_pre_net.onnx", "joint_pre_net", "joint_pre_net"),
        ]

        for onnx_file, output_dir, model_name in core_models:
            self.stats['total'] += 1
            self.convert_model(onnx_file, output_dir, model_name)

        print()
        print("Phase 2: Language-Specific Models")
        print("-" * 70)

        # Language-specific models
        languages = [
            "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok",
            "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa",
            "sa", "sat", "sd", "ta", "te", "ur"
        ]

        for lang in languages:
            onnx_file = f"joint_post_net_{lang}.onnx"
            output_dir = f"joint_post_net_{lang}"
            model_name = f"joint_post_net_{lang}"

            self.stats['total'] += 1
            self.convert_model(onnx_file, output_dir, model_name)

        elapsed = time.time() - start_time

        print()
        print("=" * 70)
        print("Conversion Summary")
        print("=" * 70)
        print()
        print(f"Total models:     {self.stats['total']}")
        print(f"Successfully converted: {self.stats['success']}")
        print(f"Skipped (exists): {self.stats['skipped']}")
        print(f"Failed:           {self.stats['failed']}")
        print(f"Time elapsed:     {elapsed:.1f} seconds")
        print()

        if self.stats['failed'] == 0:
            print("Status: ALL CONVERSIONS SUCCESSFUL")
        else:
            print("Status: SOME CONVERSIONS FAILED")
            return False

        # List output directory structure
        print()
        print("Output Directory Structure:")
        print("-" * 70)

        for item in sorted(self.output_dir.iterdir()):
            if item.is_dir():
                xml_files = list(item.glob("*.xml"))
                bin_files = list(item.glob("*.bin"))
                if xml_files and bin_files:
                    xml_size = sum(f.stat().st_size for f in xml_files) / (1024*1024)
                    bin_size = sum(f.stat().st_size for f in bin_files) / (1024*1024)
                    print(f"  {item.name:30} {xml_size:6.2f}MB + {bin_size:6.2f}MB")

        total_size = sum(
            f.stat().st_size for f in self.output_dir.rglob("*") if f.is_file()
        ) / (1024*1024*1024)

        print()
        print(f"Total OpenVINO models size: {total_size:.2f} GB")
        print()

        return True

def main():
    converter = ModelConverter()
    success = converter.convert_all()

    if success:
        print("=" * 70)
        print("Next Steps:")
        print("=" * 70)
        print()
        print("1. Test inference with OpenVINO models")
        print("2. Compare accuracy with ONNX models")
        print("3. Benchmark performance")
        print("4. Optimize further (INT8 quantization)")
        print()
        sys.exit(0)
    else:
        print()
        print("Some conversions failed. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
