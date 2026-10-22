"""Environment capture written next to every run (no secrets, no environment variables)."""
from __future__ import annotations

import platform
import subprocess
from importlib import metadata


def _v(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def capture(root=None) -> dict:
    import torch

    info = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: _v(p) for p in ("torch", "transformers", "tokenizers", "scikit-learn", "numpy", "pandas", "scipy", "datasets", "huggingface-hub")},
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_threads": torch.get_num_threads(),
    }
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip() or None
        info["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        info["git_commit"] = None
    return info
