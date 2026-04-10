#!/usr/bin/env python3
"""
Download all files from Indic Conformer HuggingFace repository
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import json

# Configuration
REPO_ID = "ai4bharat/indic-conformer-600m-multilingual"
TOKEN_FILE = "hugging_face_read_token.txt"
BASE_DIR = Path(".")
ASSETS_DIR = BASE_DIR / "assets"

# Read HF token
with open(TOKEN_FILE, 'r') as f:
    HF_TOKEN = f.read().strip()

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def get_file_list():
    """Get list of all files in the repository"""
    print("Fetching file list from HuggingFace API...")

    # Get main directory files
    url = f"https://huggingface.co/api/models/{REPO_ID}/tree/main"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    main_files = response.json()

    # Get assets directory files
    url = f"https://huggingface.co/api/models/{REPO_ID}/tree/main/assets"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    assets_files = response.json()

    return main_files, assets_files

def download_file(file_path, output_path):
    """Download a single file with progress bar"""
    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{file_path}"

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream download with progress bar
    response = requests.get(url, headers=HEADERS, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=output_path.name, leave=False) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def format_size(size_bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    print("=" * 70)
    print("Downloading Indic Conformer 600M Multilingual Model")
    print("=" * 70)
    print()

    # Create directories
    ASSETS_DIR.mkdir(exist_ok=True)

    # Get file lists
    main_files, assets_files = get_file_list()

    # Prepare download list
    downloads = []

    # Main directory files
    for item in main_files:
        if item['type'] == 'file' and item['path'] != '.gitattributes':
            downloads.append({
                'path': item['path'],
                'size': item.get('size', 0),
                'output': BASE_DIR / item['path']
            })

    # Assets directory files
    for item in assets_files:
        if item['type'] == 'file':
            downloads.append({
                'path': item['path'],
                'size': item.get('size', 0),
                'output': BASE_DIR / item['path']
            })

    # Calculate total size
    total_size = sum(d['size'] for d in downloads)

    print(f"Files to download: {len(downloads)}")
    print(f"Total size: {format_size(total_size)}")
    print()

    # Download each file
    for i, item in enumerate(downloads, 1):
        file_path = item['path']
        output_path = item['output']
        file_size = item['size']

        # Check if file already exists and has correct size
        if output_path.exists():
            existing_size = output_path.stat().st_size
            if existing_size == file_size:
                print(f"[{i}/{len(downloads)}] OK {file_path} (already exists)")
                continue

        print(f"[{i}/{len(downloads)}] Downloading {file_path} ({format_size(file_size)})...")

        try:
            download_file(file_path, output_path)
            print(f"[{i}/{len(downloads)}] OK {file_path}")
        except Exception as e:
            print(f"[{i}/{len(downloads)}] ERROR {file_path} - Error: {e}")
            continue

    print()
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)

    # List downloaded files by category
    onnx_files = list(ASSETS_DIR.glob("*.onnx"))
    other_files = [f for f in ASSETS_DIR.iterdir() if f.suffix != '.onnx']
    main_files = [f for f in BASE_DIR.iterdir() if f.is_file() and f.name != 'download_all.py']

    print(f"\nONNX Models ({len(onnx_files)} files):")
    onnx_size = sum(f.stat().st_size for f in onnx_files)
    for f in sorted(onnx_files):
        print(f"  - {f.name:50} {format_size(f.stat().st_size):>12}")
    print(f"  Total ONNX size: {format_size(onnx_size)}")

    print(f"\nAssets Files ({len(other_files)} files):")
    for f in sorted(other_files):
        if f.is_file():
            print(f"  - {f.name:50} {format_size(f.stat().st_size):>12}")

    print(f"\nMain Directory Files:")
    for f in sorted(main_files):
        if f.name not in ['hugging_face_read_token.txt', 'OpenVINO_Conversion_Strategy.md']:
            print(f"  - {f.name:50} {format_size(f.stat().st_size):>12}")

    print()
    print("Download complete!")
    print()

if __name__ == '__main__':
    main()
