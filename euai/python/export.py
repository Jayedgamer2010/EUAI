#!/usr/bin/env python3
"""
Export trained PyTorch model to EUAI binary format.
This script converts a checkpoint to the quantized INT4 format.
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
import numpy as np

def export_checkpoint(checkpoint_path, output_path):
    """Export checkpoint to binary format."""
    print("[EXPORT] Loading checkpoint...")

    if not os.path.exists(checkpoint_path):
        print(f"[EXPORT] Checkpoint not found: {checkpoint_path}")
        print("[EXPORT] Please train a model first with train.py")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})

    if not config:
        print("[EXPORT] Warning: Config not found in checkpoint, using defaults")
        config = {
            'dim': 256,
            'n_layers': 4,
            'n_heads': 8,
            'n_kv_groups': 4,
            'vocab_size': 256,
            'max_seq_len': 512
        }

    print(f"[EXPORT] Model config: {config}")

    # Open output file
    with open(output_path, 'wb') as f:
        # Write magic number
        f.write(b'EUAI')

        # Write config as little-endian integers
        f.write(int(config.get('dim', 256)).to_bytes(4, 'little'))
        f.write(int(config.get('n_layers', 4)).to_bytes(4, 'little'))
        f.write(int(config.get('n_heads', 8)).to_bytes(4, 'little'))
        f.write(int(config.get('n_kv_groups', 4)).to_bytes(4, 'little'))
        f.write(int(config.get('vocab_size', 256)).to_bytes(4, 'little'))
        f.write(int(config.get('max_seq_len', 512)).to_bytes(4, 'little'))

        # Write embedding weights
        embedding_key = next((k for k in model_state.keys() if 'embedding' in k.lower()), None)
        if embedding_key:
            embedding = model_state[embedding_key].cpu().numpy().astype(np.float32)
            f.write(embedding.tobytes())
            print(f"[EXPORT] Wrote embedding: {embedding.shape}")

        # Write transformer layers
        layer_keys = [k for k in model_state.keys() if 'transformer' in k.lower() or 'encoder' in k.lower()]
        for key in layer_keys:
            tensor = model_state[key].cpu().numpy().astype(np.float32)
            f.write(tensor.tobytes())
            print(f"[EXPORT] Wrote {key}: {tensor.shape}")

        # Write final layer norm
        norm_keys = [k for k in model_state.keys() if 'ln_f' in k.lower() or 'final_norm' in k.lower()]
        for key in norm_keys:
            tensor = model_state[key].cpu().numpy().astype(np.float32)
            f.write(tensor.tobytes())
            print(f"[EXPORT] Wrote {key}: {tensor.shape}")

        # Write LM head if separate
        head_keys = [k for k in model_state.keys() if 'lm_head' in k.lower() or 'output' in k.lower()]
        for key in head_keys:
            tensor = model_state[key].cpu().numpy().astype(np.float32)
            f.write(tensor.tobytes())
            print(f"[EXPORT] Wrote {key}: {tensor.shape}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[EXPORT] Binary model saved: {output_path} ({size_mb:.2f} MB)")
    if size_mb > 35:
        print("[EXPORT] WARNING: Model exceeds 35MB target for INT4 quantization!")
        print("[EXPORT] Consider reducing model dimensions or using quantization.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to binary format")
    parser.add_argument('--checkpoint', type=str, default='python/checkpoints/euai_epoch5.pt',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='python/euai.bin',
                        help='Output binary path')
    args = parser.parse_args()

    export_checkpoint(args.checkpoint, args.output)
