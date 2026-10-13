"""Configuration loading with strict defaults."""
from __future__ import annotations

import copy

import yaml

DEFAULTS = {
    "experiment": {"id": "M1", "name": "", "seed": 42, "description": ""},
    "data": {
        "images_dir": "data",
        "split_csv": "data/processed/splits_v1.csv",
        "label_col": "kl_label",
    },
    "preprocessing": {
        "image_size": [224, 224],
        "contrast": "none",
        "clahe_clip": 2.0,
        "clahe_grid": [8, 8],
        "normalize": "raw",
        "cache_dir": "data/processed/cache",
    },
    "roi": {"enabled": False, "margin_h": 0.18, "margin_w": 0.22},
    "model": {
        "architecture": "standard_cnn",
        "pretrained": False,
        "dropout": 0.3,
        "num_classes": 5,
    },
    "training": {
        "batch_size": 32,
        "epochs": 40,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "early_stop_patience": 8,
        "class_weighting": "weighted_ce",
        "augmentation": "light",
        "workers": 0,
        "device": "auto",
        "precision": "fp32",
    },
    "eval": {"test_role": "test"},
    "compute": {"throughput_batch": 32, "latency_repeats": 50, "latency_warmup": 10},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return deep_merge(DEFAULTS, raw)


def resolve_paths(cfg: dict, project_root: str) -> dict:
    """Make data/split paths absolute relative to the project root."""
    import os
    root = os.path.abspath(project_root)
    for key in ("split_csv", "images_dir"):
        p = cfg["data"].get(key)
        if p and not os.path.isabs(p):
            cfg["data"][key] = os.path.join(root, p)
    p = cfg["preprocessing"].get("cache_dir")
    if p and not os.path.isabs(p):
        cfg["preprocessing"]["cache_dir"] = os.path.join(root, p)
    return cfg
