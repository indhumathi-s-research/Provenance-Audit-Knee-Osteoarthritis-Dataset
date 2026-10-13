"""Shared utilities: hashing, env/git info, device selection, seeding."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Fixed image resolutions make cudnn autotuning valid; it is ~1.3x faster
    # on this Pascal GPU than deterministic mode. Weight init and data order
    # remain seeded for reproducibility.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(precision: str = "fp32") -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_commit(repo_root: Path | None = None) -> str:
    repo_root = repo_root or Path.cwd()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "no-git"


def git_branch(repo_root: Path | None = None) -> str:
    repo_root = repo_root or Path.cwd()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "detached"
    except Exception:
        pass
    return "unknown"


def hardware_info() -> dict:
    info = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        )
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu"] = "none"
    info["torch"] = torch.__version__
    info["torchvision"] = getattr(torch, "__version__", "unknown")
    return info


def env_info() -> dict:
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware_info(),
    }


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.2f} GB"
