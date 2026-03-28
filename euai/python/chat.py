#!/usr/bin/env python3
"""
Interactive chat interface for EUAI.
Uses the C++ binary via subprocess or loads PyTorch model directly.
"""

import os
import sys
import subprocess
import argparse

def chat_cpp(model_path, config_dir, max_tokens):
    """Use C++ binary for inference."""
    binary = "./euai"
    if not os.path.exists(binary):
        print(f"[CHAT] Error: C++ binary not found at {binary}")
        print("[CHAT] Please build the C++ code first: g++ -std=c++17 -O2 -o euai ...")
        sys.exit(1)

    print(f"[CHAT] Starting C++ inference with model: {model_path}")
    proc = subprocess.Popen(
        [binary, "--model", model_path, "--config", config_dir, "--max-tokens", str(max_tokens)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("\n=== EUAI Chat (C++ backend) ===")
    print("Type your message (Ctrl+D or 'quit' to exit)\n")

    try:
        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() in ['quit', 'exit']:
                    break
                if not user_input.strip():
                    continue

                # Send input and get response
                proc.stdin.write(user_input + "\n")
                proc.stdin.flush()
                # Note: This is simplified - actual C++ binary runs interactively
                # For proper subprocess interaction would need different approach
                print("\n[CHAT] Running in subprocess mode. Please use ./euai directly for interactive mode.")
                break
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\n[CHAT] Exiting...")
    finally:
        proc.terminate()

def chat_pytorch(model_path):
    """Load PyTorch model and run inference."""
    print("[CHAT] PyTorch inference not yet implemented")
    print("[CHAT] Use the C++ binary or complete the implementation")

def main():
    parser = argparse.ArgumentParser(description="EUAI Chat Interface")
    parser.add_argument('--model', type=str, default='python/euai.bin',
                        help='Model binary path')
    parser.add_argument('--config', type=str, default='config',
                        help='Config directory')
    parser.add_argument('--max-tokens', type=int, default=200,
                        help='Maximum tokens to generate')
    parser.add_argument('--backend', choices=['cpp', 'pytorch'], default='cpp',
                        help='Backend to use')
    args = parser.parse_args()

    if args.backend == 'cpp':
        chat_cpp(args.model, args.config, args.max_tokens)
    else:
        chat_pytorch(args.model)

if __name__ == "__main__":
    main()
