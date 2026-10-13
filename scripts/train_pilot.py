"""PHASES 3-7 — Train one pilot configuration end-to-end with tracking.

Usage:
    python scripts/train_pilot.py --config configs/m2.yaml \
        --splits data/processed/splits_v1.csv \
        --runs-dir runs \
        [--skip-compute]

Saves per run:
    config.json, run_metadata.json, compute.json, checkpoints/best_model.pt,
    history.csv, val_metrics.json, test_metrics.json, predictions_test.csv,
    predictions_val.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import compute as compute_mod
from src.config import load_config, resolve_paths
from src.models import build_model
from src.tracking import create_run_dir, save_checkpoint, save_run_metadata
from src.train_eval import build_loaders, evaluate, train_model
from src.utils import (
    get_device,
    save_json,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_split_version(csv_path: str) -> str:
    import hashlib
    df = pd.read_csv(csv_path, usecols=["image_path", "role"])
    text = ",".join(
        df.sort_values("image_path").apply(lambda r: f"{r['image_path']}|{r['role']}", axis=1).tolist()
    )
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def log(msg: str) -> None:
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default=str(PROJECT_ROOT / "data" / "processed" / "splits_v1.csv"))
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--skip-compute", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="Override epoch budget")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    ap.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test limit per epoch")
    ap.add_argument("--dataset-version-file", default=str(PROJECT_ROOT / "data" / "processed" / "dataset_version.txt"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    resolve_paths(cfg, str(PROJECT_ROOT))
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.max_train_batches:
        cfg["training"]["max_train_batches"] = args.max_train_batches
    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    device = get_device()
    log(f"device: {device} | exp: {cfg['experiment']['id']} | arch: {cfg['model']['architecture']}")
    log(f"epochs={cfg['training']['epochs']} batch={cfg['training']['batch_size']} "
        f"workers={cfg['training']['workers']} lr={cfg['training']['lr']}")

    split_df = pd.read_csv(args.splits)
    dataset_version = Path(args.dataset_version_file).read_text().strip() if Path(args.dataset_version_file).exists() else "unknown"
    split_ver = load_split_version(args.splits)

    run_dir = create_run_dir(args.runs_dir, cfg["experiment"]["id"], seed)
    save_json(cfg, run_dir / "config.json")

    t0 = time.time()
    train_loader, val_loader, test_loader, train_df, class_weights = build_loaders(
        cfg, split_df, device, seed
    )
    log(f"loaders built in {time.time()-t0:.1f}s | "
        f"train={len(train_df)} val={len(split_df[split_df.role=='val'])} "
        f"test={len(split_df[split_df.role=='test'])}")
    log(f"class weights (train only): {[round(float(w),4) for w in class_weights]}")

    model = build_model(cfg)
    model = model.to(device)
    log(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    compute_profile = None
    if not args.skip_compute:
        log("measuring compute profile (params, MACs, latency, memory) ...")
        sample_img = train_df.iloc[0]["image_path"]
        compute_profile = compute_mod.full_compute_profile(cfg, sample_img, device)
        compute_profile["precision"] = cfg["training"]["precision"]
        compute_profile["image_resolution"] = list(cfg["preprocessing"]["image_size"])
        compute_profile["hardware"] = str(device)
        save_json(compute_profile, run_dir / "compute.json")
        log(f"  params={compute_profile['parameters']:,}  MACs={compute_profile['macs']:,}  "
            f"lat_b1={compute_profile['inference_latency_batch1_ms']['mean_ms']:.2f}ms")

    log("starting training ...")
    result = train_model(cfg, model, train_loader, val_loader, class_weights, device, run_dir)
    log(f"training done: {result['epochs_run']} epochs, {result['training_time_s']:.1f}s, "
        f"best val macro-F1={result['best_val_macro_f1']:.4f}")
    model.load_state_dict(result["best_state"])

    log("evaluating on val ...")
    val_metrics, val_pred = evaluate(model, val_loader, device)
    log("evaluating on test ...")
    test_metrics, test_pred = evaluate(model, test_loader, device)
    save_json(val_metrics, run_dir / "val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    val_pred.to_csv(run_dir / "predictions_val.csv", index=False)
    test_pred.to_csv(run_dir / "predictions_test.csv", index=False)

    save_checkpoint(run_dir, model, result["best_state"], result["history"], test_metrics)
    save_run_metadata(
        run_dir,
        cfg,
        dataset_version,
        split_ver,
        result["epochs_run"],
        extra={
            "val_metrics_summary": {
                "macro_f1": val_metrics["macro_f1"],
                "qwk": val_metrics["qwk"],
                "balanced_accuracy": val_metrics["balanced_accuracy"],
            },
            "test_metrics_summary": {
                "macro_f1": test_metrics["macro_f1"],
                "qwk": test_metrics["qwk"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
            },
            "training_time_s": result["training_time_s"],
        },
    )

    print("\n===== RESULTS =====", flush=True)
    for name, m in (("VAL", val_metrics), ("TEST", test_metrics)):
        print(f"{name} | acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"bAcc={m['balanced_accuracy']:.4f} QWK={m['qwk']:.4f}", flush=True)
    print(f"training_time_s={result['training_time_s']:.1f}", flush=True)
    print(f"run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
