"""Shared hyperparameter search space for the optimization stage.

Every configuration is a point in a continuous box x in [0,1]^D. The same box,
decoder and objective are used by Random Search, PSO and SOA so the three
methods are compared on an identical problem.

Base configuration: M6 (ResNet-18 ImageNet-pretrained, CLAHE, ImageNet norm).
Only hyperparameters are searched; architecture/data/imbalance handling are fixed
to preserve the protocol (weighted CE from train only, patient-disjoint split).

Dimensions (D = 8):
    0  lr          log-uniform [1e-4, 3e-3]
    1  weight_decay log-uniform [1e-5, 1e-3]
    2  dropout     uniform     [0.1, 0.5]
    3  clahe_clip  uniform     [1.0, 4.0]
    4  augmentation categorical ['light', 'none']
    5  scheduler    categorical ['cosine', 'none']
    6  optimizer    categorical ['adam', 'sgd']
    7  batch_size   categorical [16, 32, 64]
"""
from __future__ import annotations

import numpy as np

DIM = 8

# (kind, lo, hi) for continuous dims; (kind, options) for categorical dims
_SPECS = [
    ("log", 1e-4, 3e-3),
    ("log", 1e-5, 1e-3),
    ("lin", 0.1, 0.5),
    ("lin", 1.0, 4.0),
    ("cat", ["light", "none"]),
    ("cat", ["cosine", "none"]),
    ("cat", ["adam", "sgd"]),
    ("cat", [16, 32, 64]),
]

# lower/upper bounds used by PSO/SOA (all normalized to [0,1])
LB = np.zeros(DIM)
UB = np.ones(DIM)


def _cat(x, options):
    idx = int(np.clip(round(float(x) * (len(options) - 1)), 0, len(options) - 1))
    return options[idx]


def decode(vec) -> dict:
    """Map a continuous box point to config overrides (merged onto M6 base)."""
    v = np.asarray(vec, dtype=float)
    lr = 10 ** (np.log10(_SPECS[0][1]) + v[0] * (np.log10(_SPECS[0][2]) - np.log10(_SPECS[0][1])))
    wd = 10 ** (np.log10(_SPECS[1][1]) + v[1] * (np.log10(_SPECS[1][2]) - np.log10(_SPECS[1][1])))
    dropout = _SPECS[2][1] + v[2] * (_SPECS[2][2] - _SPECS[2][1])
    clahe_clip = _SPECS[3][1] + v[3] * (_SPECS[3][2] - _SPECS[3][1])
    augment = _cat(v[4], _SPECS[4][1])
    scheduler = _cat(v[5], _SPECS[5][1])
    optimizer = _cat(v[6], _SPECS[6][1])
    batch_size = _cat(v[7], _SPECS[7][1])
    return {
        "training": {
            "lr": float(lr),
            "weight_decay": float(wd),
            "augmentation": augment,
            "scheduler": scheduler,
            "optimizer": optimizer,
            "batch_size": int(batch_size),
        },
        "model": {"dropout": float(dropout)},
        "preprocessing": {"clahe_clip": float(clahe_clip)},
    }


def random_vector(rng: np.random.Generator) -> np.ndarray:
    return rng.random(DIM)


def human_readable(vec) -> dict:
    """Decode to a human-friendly dict (for logging/reports)."""
    ov = decode(vec)
    t = ov["training"]
    return {
        "lr": t["lr"],
        "weight_decay": t["weight_decay"],
        "dropout": ov["model"]["dropout"],
        "clahe_clip": ov["preprocessing"]["clahe_clip"],
        "augmentation": t["augmentation"],
        "scheduler": t["scheduler"],
        "optimizer": t["optimizer"],
        "batch_size": t["batch_size"],
    }


def vector_key(vec, nd: int = 6) -> tuple:
    return tuple(round(float(x), nd) for x in vec)
