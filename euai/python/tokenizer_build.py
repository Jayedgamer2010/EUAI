#!/usr/bin/env python3
"""
BPE tokenizer training for EUAI.
Trains a tokenizer on the prepared training data.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def build_tokenizer(train_data_path, output_dir):
    """Build BPE tokenizer from training data."""
    print("[TOKENIZER_BUILD] Building tokenizer...")

    os.makedirs(output_dir, exist_ok=True)

    # Load training data
    print(f"[TOKENIZER_BUILD] Loading data from {train_data_path}")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    # Simple character-level vocabulary for demo
    # In production, would use proper BPE algorithm
    vocab = set()
    for text in texts:
        vocab.update(list(text))

    # Add special tokens
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
    for tok in special_tokens:
        vocab.add(tok)

    # Create vocab mapping (char -> id)
    vocab = sorted(list(vocab))
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}
    vocab_rev = {idx: token for token, idx in vocab_dict.items()}

    # Save vocab.json
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
    print(f"[TOKENIZER_BUILD] Saved vocab with {len(vocab)} tokens to {vocab_path}")

    # Create minimal merges (empty for char-level)
    merges_path = os.path.join(output_dir, 'merges.json')
    with open(merges_path, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2)
    print(f"[TOKENIZER_BUILD] Saved merges to {merges_path}")

    print("[TOKENIZER_BUILD] Tokenizer build complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BPE tokenizer for EUAI")
    parser.add_argument('--data', type=str, default='euai/python/data/train.txt',
                        help='Training data path')
    parser.add_argument('--output-dir', type=str, default='config',
                        help='Output directory for tokenizer files')
    args = parser.parse_args()

    build_tokenizer(args.data, args.output_dir)
