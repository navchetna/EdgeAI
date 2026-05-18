#!/usr/bin/env python3
"""
Convert Supertonic TTS ONNX models to OpenVINO IR format
Supports CPU/GPU acceleration and FP16/FP32 precision
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

# Default paths
REPO_URL = "https://huggingface.co/Supertone/supertonic-3"
DEFAULT_REPO_DIR = "supertonic-3"
DEFAULT_ONNX_SUBDIR = "onnx"


def ensure_repo_exists(repo_dir: str = DEFAULT_REPO_DIR) -> Path:
    """
    Ensure the Supertonic repository exists, clone if missing.

    Args:
        repo_dir: Path to the repository directory

    Returns:
        Path to the repository
    """
    repo_path = Path(repo_dir)

    if not repo_path.exists():
        print(f"Repository not found at '{repo_dir}'. Cloning from Hugging Face...")
        print(f"  URL: {REPO_URL}")
        print()

        try:
            result = subprocess.run(
                ["git", "clone", REPO_URL, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for large repo
            )

            if result.returncode != 0:
                print(f"ERROR: Failed to clone repository")
                print(f"  {result.stderr}")
                raise RuntimeError("Failed to clone Supertonic repository")

            print(f"Repository cloned successfully to '{repo_dir}'")
            print()

        except FileNotFoundError:
            print("ERROR: 'git' command not found. Please install git.")
            raise
        except subprocess.TimeoutExpired:
            print("ERROR: Clone operation timed out.")
            raise

    return repo_path


class SupertonicModelConverter:
    """Convert Supertonic TTS ONNX models to OpenVINO IR format"""

    # ONNX model names used by Supertonic TTS
    MODEL_NAMES = [
        "duration_predictor",
        "text_encoder",
        "vector_estimator",
        "vocoder",
    ]

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        precision: str = "fp16",
        device: str = "CPU",
    ):
        """
        Initialize the converter.

        Args:
            input_dir: Directory containing ONNX models
            output_dir: Output directory for OpenVINO IR models
            precision: Precision format - 'fp16' or 'fp32'
            device: Target device - 'CPU' or 'GPU'
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.precision = precision.lower()
        self.device = device.upper()

        # Validate precision
        if self.precision not in ("fp16", "fp32"):
            raise ValueError(f"Invalid precision: {precision}. Must be 'fp16' or 'fp32'")

        # Validate device
        if self.device not in ("CPU", "GPU"):
            raise ValueError(f"Invalid device: {device}. Must be 'CPU' or 'GPU'")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }

    def convert_model(self, model_name: str) -> bool:
        """
        Convert a single ONNX model to OpenVINO IR.

        Args:
            model_name: Name of the model (without extension)

        Returns:
            True if successful, False otherwise
        """
        onnx_path = self.input_dir / f"{model_name}.onnx"
        xml_path = self.output_dir / f"{model_name}.xml"
        bin_path = self.output_dir / f"{model_name}.bin"

        self.stats["total"] += 1

        # Check if ONNX model exists
        if not onnx_path.exists():
            print(f"  SKIP - {model_name}.onnx (not found)")
            self.stats["skipped"] += 1
            return False

        # Check if already converted
        if xml_path.exists() and bin_path.exists():
            print(f"  SKIP - {model_name} (already exists)")
            self.stats["skipped"] += 1
            return True

        # Build conversion command using OpenVINO Model Converter (ovc)
        cmd = [
            "ovc",
            str(onnx_path),
            "--output_model",
            str(self.output_dir / model_name),
        ]

        # Add FP16 compression flag if requested
        if self.precision == "fp16":
            cmd.append("--compress_to_fp16")

        try:
            precision_str = self.precision.upper()
            print(f"  Converting {model_name} ({precision_str})...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode == 0:
                # Verify output files exist
                if xml_path.exists() and bin_path.exists():
                    xml_size = xml_path.stat().st_size / (1024 * 1024)
                    bin_size = bin_path.stat().st_size / (1024 * 1024)
                    print(f"  OK   - {model_name} ({xml_size:.2f}MB + {bin_size:.2f}MB)")
                    self.stats["success"] += 1
                    return True
                else:
                    print(f"  FAIL - {model_name} (output files not created)")
                    self.stats["failed"] += 1
                    return False
            else:
                print(f"  FAIL - {model_name}")
                if result.stderr:
                    print(f"       Error: {result.stderr[:300]}")
                self.stats["failed"] += 1
                return False

        except subprocess.TimeoutExpired:
            print(f"  FAIL - {model_name} (timeout)")
            self.stats["failed"] += 1
            return False
        except Exception as e:
            print(f"  FAIL - {model_name} ({e})")
            self.stats["failed"] += 1
            return False

    def convert_all(self) -> dict:
        """
        Convert all Supertonic TTS models.

        Returns:
            Dictionary with conversion statistics
        """
        print("=" * 70)
        print("Converting Supertonic TTS ONNX Models to OpenVINO IR Format")
        print("=" * 70)
        print()
        print(f"  Input directory:  {self.input_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Precision:        {self.precision.upper()}")
        print(f"  Target device:    {self.device}")
        print()
        print("-" * 70)

        start_time = time.time()

        for model_name in self.MODEL_NAMES:
            self.convert_model(model_name)

        elapsed = time.time() - start_time

        print()
        print("=" * 70)
        print("Conversion Summary")
        print("-" * 70)
        print(f"  Total models:   {self.stats['total']}")
        print(f"  Successful:     {self.stats['success']}")
        print(f"  Failed:         {self.stats['failed']}")
        print(f"  Skipped:        {self.stats['skipped']}")
        print(f"  Time elapsed:   {elapsed:.2f} seconds")
        print("=" * 70)

        return self.stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Supertonic TTS ONNX models to OpenVINO IR format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo-dir",
        "-r",
        type=str,
        default=DEFAULT_REPO_DIR,
        help="Supertonic repository directory (will clone if not exists)",
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=None,
        help=f"Directory containing ONNX models (default: <repo-dir>/{DEFAULT_ONNX_SUBDIR})",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for OpenVINO models (default: ov_model_{precision})",
    )

    parser.add_argument(
        "--precision",
        "-p",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Model precision format",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["CPU", "GPU"],
        default="CPU",
        help="Target accelerator device",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure repository exists
    repo_path = ensure_repo_exists(args.repo_dir)

    # Set default input directory based on repo path if not specified
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = repo_path / DEFAULT_ONNX_SUBDIR

    # Set default output directory based on precision if not specified
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"ov_model_{args.precision}"

    # Create converter and run
    converter = SupertonicModelConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        precision=args.precision,
        device=args.device,
    )

    stats = converter.convert_all()

    # Return non-zero exit code if any conversions failed
    if stats["failed"] > 0:
        exit(1)


if __name__ == "__main__":
    main()
