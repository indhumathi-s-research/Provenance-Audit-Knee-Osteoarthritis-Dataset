"""PHASE 6 — Standalone computational measurement for a pilot configuration.

Usage:
    python scripts/measure_compute.py --config configs/m2.yaml --image data/test/0/9157453L.png

Measures (batch-1, warm-up, repeats, CUDA-synced): preprocessing latency,
CNN inference latency, throughput, parameters, MACs/FLOPs, model size,
peak memory (inference + training). Records hardware, software, precision
and image resolution. Saves compute.json to the given --out file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import compute as compute_mod
from src.config import load_config, resolve_paths
from src.models import build_model
from src.utils import env_info, get_device, save_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", default=str(PROJECT_ROOT / "data" / "test" / "0" / "9157453L.png"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    resolve_paths(cfg, str(PROJECT_ROOT))
    cfg["compute"]["latency_repeats"] = args.repeats
    cfg["compute"]["latency_warmup"] = args.warmup

    device = get_device()
    t0 = time.time()
    profile = compute_mod.full_compute_profile(cfg, args.image, device)
    profile["precision"] = cfg["training"]["precision"]
    profile["image_resolution"] = list(cfg["preprocessing"]["image_size"])
    profile["hardware"] = env_info()["hardware"]
    profile["elapsed_s"] = round(time.time() - t0, 1)

    out = args.out or str(
        PROJECT_ROOT / "reports" / f"compute_{cfg['experiment']['id']}.json"
    )
    save_json(profile, out)

    print(json.dumps(profile, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
