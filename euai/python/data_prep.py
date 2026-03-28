#!/usr/bin/env python3
"""
Data preparation for EUAI training.
Downloads and processes training data.
"""

import os
import sys
import argparse
from pathlib import Path

def prepare_data(output_path):
    """Prepare training data."""
    print("[DATA_PREP] Preparing training data...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if user has data in expected location
    source_path = Path("data/train.txt")
    if source_path.exists():
        print(f"[DATA_PREP] Copying from {source_path} to {output_path}")
        import shutil
        shutil.copy(source_path, output_path)
    else:
        # Create minimal synthetic data for demo
        print(f"[DATA_PREP] Creating synthetic training data at {output_path}")
        samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets.",
            "Python is a popular programming language.",
            "C++ provides high performance for systems programming.",
            "Hello how are you today?",
            "What is the capital of France? Paris.",
            "2 plus 2 equals 4.",
            "The sky is blue during the day.",
            "Neural networks have multiple layers."
        ] * 100  # Repeat to have enough data

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in samples:
                f.write(line + '\n')

    print(f"[DATA_PREP] Training data ready: {output_path}")
    print(f"[DATA_PREP] Lines: {sum(1 for _ in open(output_path, 'r', encoding='utf-8'))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for EUAI")
    parser.add_argument('--output', type=str, default='euai/python/data/train.txt',
                        help='Output path for training data')
    args = parser.parse_args()

    prepare_data(args.output)
