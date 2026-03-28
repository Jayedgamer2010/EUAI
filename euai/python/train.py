#!/usr/bin/env python3
"""
Training script for EUAI language model using PyTorch.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from pathlib import Path
import signal
import psutil

class EUAIDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.data = []
        self.max_len = max_len
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = tokenizer.encode(line)
                    if len(tokens) > max_len:
                        tokens = tokens[:max_len]
                    self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=4, n_heads=8, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, batch_first=True, dim_feedforward=dim*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight  # Tie weights

    def forward(self, input_ids):
        x = self.embedding(input_ids) + self.pos_encoding[:, :input_ids.size(1), :]
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def check_memory():
    """Check RAM usage and warn if exceeding threshold."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    if mem_mb > 580:
        print(f"[WARNING] RAM usage: {mem_mb:.1f} MB (exceeds 580 MB threshold)")
        return False
    return True

def train_model(args):
    print("[TRAIN] Starting training...")

    # Load tokenizer
    with open(os.path.join(args.config, 'vocab.json'), 'r') as f:
        vocab_dict = json.load(f)
    vocab_size = len(vocab_dict)

    # Simple char-level tokenizer wrapper
    class CharTokenizer:
        def __init__(self, vocab_dict):
            self.stoi = vocab_dict
            self.itos = {v: k for k, v in vocab_dict.items()}
        def encode(self, text):
            return [self.stoi.get(c, self.stoi['<unk>']) for c in text]
        def decode(self, ids):
            return ''.join([self.itos.get(i, '<unk>') for i in ids])

    tokenizer = CharTokenizer(vocab_dict)

    # Dataset and DataLoader
    dataset = EUAIDataset(args.data, tokenizer, max_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TRAIN] Using device: {device}")

    model = SimpleTransformer(
        vocab_size=vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        max_len=args.seq_len
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    total_steps = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_steps += 1

            if batch_idx % 10 == 0:
                print(f"  Step {batch_idx}, Loss: {loss.item():.4f}, RAM: {psutil.Process().memory_info().rss/1024/1024:.1f} MB")

            # Overfit test for debugging
            if args.overfit_test and total_steps >= 50:
                print("[TRAIN] Overfit test completed, stopping.")
                break

        print(f"[TRAIN] Epoch {epoch+1}, Average Loss: {epoch_loss/len(dataloader):.4f}")

        # Memory check per epoch
        if not check_memory():
            break

    # Save checkpoint
    checkpoint_dir = Path('python/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f'euai_epoch{epoch+1}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'dim': args.dim,
            'n_layers': args.layers,
            'n_heads': args.heads,
            'max_seq_len': args.seq_len
        }
    }, checkpoint_path)
    print(f"[TRAIN] Checkpoint saved: {checkpoint_path}")

    # Export to C++ binary format
    export_model(model, checkpoint_path, args.export)

def export_model(model, checkpoint_path, export_path):
    """Export PyTorch model to EUAI binary format."""
    print("[EXPORT] Exporting model to binary format...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    config = checkpoint['config']

    # Create binary format
    with open(export_path, 'wb') as f:
        # Magic number
        f.write(b'EUAI')

        # Write config
        f.write(config['dim'].to_bytes(4, 'little'))
        f.write(config['n_layers'].to_bytes(4, 'little'))
        f.write(config['n_heads'].to_bytes(4, 'little'))
        f.write(config['n_layers'].to_bytes(4, 'little'))  # n_kv_groups = n_layers for simplicity
        f.write(config['vocab_size'].to_bytes(4, 'little'))
        f.write(config['max_seq_len'].to_bytes(4, 'little'))

        # Write weights (simplified)
        # In real implementation would write each tensor with shape and quantize
        for name, param in model.named_parameters():
            if 'weight' in name:
                data = param.detach().cpu().numpy().astype('float32')
                f.write(data.tobytes())

    print(f"[EXPORT] Model exported to {export_path}")
    print(f"[EXPORT] Size: {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EUAI language model")
    parser.add_argument('--data', type=str, default='euai/python/data/train.txt',
                        help='Training data path')
    parser.add_argument('--config', type=str, default='config',
                        help='Config directory with vocab.json')
    parser.add_argument('--dim', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--overfit-test', action='store_true',
                        help='Run overfit test (short training)')
    parser.add_argument('--export', type=str, default='euai.bin',
                        help='Export path for binary model')
    args = parser.parse_args()

    train_model(args)
