"""Experiment tracking: run dirs, metadata, checkpoints, predictions."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import torch

from src.utils import env_info, git_branch, git_commit, save_json


def create_run_dir(base: str, exp_id: str, seed: int) -> Path:
    base = Path(base)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{exp_id}_seed{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def save_run_metadata(
    run_dir: Path,
    cfg: dict,
    dataset_version: str,
    split_version: str,
    train_budget: int,
    extra: dict | None = None,
) -> None:
    meta = {
        "experiment": cfg["experiment"],
        "dataset_version": dataset_version,
        "split_version": split_version,
        "training_epoch_budget": train_budget,
        "env": env_info(),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    save_json(meta, run_dir / "run_metadata.json")


def save_checkpoint(run_dir: Path, model, best_state: dict, history: list, metrics: dict | None = None) -> Path:
    ckpt = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "history": history,
        "metrics": metrics,
    }
    path = run_dir / "checkpoints" / "best_model.pt"
    torch.save(ckpt, path)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if metrics:
        save_json(metrics, run_dir / "metrics.json")
    return path
