#!/usr/bin/env python3
"""
Runner: Calls the C++ euai binary and parses output
"""

import subprocess
import json
import os
from pathlib import Path

class Runner:
    def __init__(self, binary_path, config_dir):
        self.binary_path = Path(binary_path)
        self.config_dir = Path(config_dir)
        self.model_path = self.config_dir.parent / "models" / "qwen2.5-coder-0.5b-instruct-q2_k.gguf"

        if not self.binary_path.exists():
            raise FileNotFoundError(f"Binary not found: {binary_path}")

    def generate(self, prompt: str, max_tokens: int = 200) -> tuple[str, str]:
        """Call euai binary and return (source, response)"""
        try:
            cmd = [
                str(self.binary_path),
                f"--model={self.model_path}",
                f"--config={self.config_dir}",
                f"--max-tokens={max_tokens}"
            ]

            # Run subprocess with cwd set to project root (parent of euai dir)
            project_root = self.binary_path.parent.parent
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_root
            )

            if result.returncode != 0:
                return "ERROR", f"Binary failed: {result.stderr}"

            output = result.stdout
            # Extract source and response
            lines = output.split('\n')
            for line in lines:
                # Only match lines that start with [SOURCE: to avoid debug logs like [InferenceEngine]
                if line.startswith('[SOURCE:') and ']' in line:
                    parts = line.split(']', 1)
                    if len(parts) == 2:
                        # Extract source from inside the brackets, e.g., "[SOURCE: MATH]"
                        tag = parts[0][1:]  # remove leading '['
                        # tag is "SOURCE: MATH"
                        if tag.startswith("SOURCE: "):
                            source = tag.split(": ", 1)[1]
                        else:
                            source = "NEURAL"
                        return source, parts[1].strip()

            # No source tag found
            return "NEURAL", output.strip()

        except subprocess.TimeoutExpired:
            return "ERROR", "Generation timeout"
        except Exception as e:
            return "ERROR", f"Runner exception: {str(e)}"

    def health_check(self) -> dict:
        """Check if binary is available and model exists"""
        return {
            "status": "ok",
            "model": "euai",
            "model_loaded": self.model_path.exists(),
            "binary_ready": self.binary_path.exists() and os.access(self.binary_path, os.X_OK),
            "version": "1.0"
        }
