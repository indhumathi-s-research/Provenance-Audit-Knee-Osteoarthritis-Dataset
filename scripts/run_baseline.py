"""BASELINE STAGE — train ONE configuration into experiments/baseline/<ID>/.

Rules enforced:
    - frozen dataset KOA_KL_PILOT_v1 must exist and match (dataset, split) hashes
    - patient-disjointness of the split is verified before every experiment
    - class weights computed from TRAINING data only; weighted CE is the only
      imbalance mechanism
    - model selection uses VALIDATION macro-F1 only; the test set is NEVER loaded

Usage:
    python scripts/run_baseline.py --config configs/m2.yaml [--smoke]

Outputs to experiments/baseline/<ID>/:
    config.yaml, metrics.json (VAL), predictions.csv (VAL),
    checkpoints/best_model.pt, training_history.csv,
    computational_metrics.json, run_metadata.json
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import compute as compute_mod
from src.config import load_config, resolve_paths
from src.models import build_model, count_parameters
from src.splits import verify_no_leakage
from src.train_eval import build_loaders, evaluate, run_epoch, train_model
from src.utils import get_device, save_json, set_seed, sha256_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREEZE_PATH = PROJECT_ROOT / "data" / "processed" / "KOA_KL_PILOT_v1_freeze.json"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits_v1.csv"


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def check_frozen_dataset() -> dict:
    if not FREEZE_PATH.exists():
        raise SystemExit("frozen dataset missing — run scripts/freeze_dataset.py first")
    return __import__("json").loads(FREEZE_PATH.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "experiments" / "baseline"))
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 20 batches plumbing test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    resolve_paths(cfg, str(PROJECT_ROOT))
    exp_id = cfg["experiment"]["id"]
    seed = cfg["experiment"]["seed"]

    freeze = check_frozen_dataset()
    set_seed(seed)
    device = get_device()

    split_df = pd.read_csv(SPLITS_PATH)
    leak = verify_no_leakage(split_df)
    if leak["patients_in_multiple_roles"] > 0:
        raise SystemExit(f"ABORT: split not patient-disjoint: {leak['violations'][:3]}")
    split_ver = sha256_text(
        ",".join(
            split_df.sort_values("image_path")
            .apply(lambda r: f"{r['image_path']}|{r['role']}", axis=1)
            .tolist()
        )
    )[:12]
    if split_ver != freeze["split_version"]:
        raise SystemExit(
            f"ABORT: split version mismatch (csv={split_ver}, frozen={freeze['split_version']})"
        )
    log(f"freeze {freeze['freeze_id']} verified | patients={leak['patients']} | "
        f"disjoint=YES | device={device}")

    if args.smoke:
        cfg["training"]["epochs"] = 2
        cfg["training"]["max_train_batches"] = 20

    out_dir = Path(args.out_root) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader, val_loader, _unused_test, train_df, class_weights = build_loaders(
        cfg, split_df, device, seed
    )
    del _unused_test
    log(f"train={len(train_df)} val={len(split_df[split_df.role == 'val'])} "
        f"(test NOT loaded) | class weights (train-only): "
        f"{[round(float(w), 4) for w in class_weights]}")

    model = build_model(cfg).to(device)

    sample_img = train_df.iloc[0]["image_path"]
    cpu0 = compute_mod.peak_cpu_memory_bytes()
    t_c0 = time.perf_counter()
    comp = compute_mod.full_compute_profile(cfg, sample_img, device)
    comp["profile_wall_s"] = round(time.perf_counter() - t_c0, 2)
    save_json(comp, out_dir / "computational_metrics.json")
    log(f"compute profile: params={comp['parameters']:,} MACs={comp['macs']:,} "
        f"inf_b1={comp['inference_latency_batch1_ms']['mean_ms']:.2f}ms "
        f"e2e_b1={comp['end_to_end_latency_batch1_ms']['mean_ms']:.2f}ms")

    result = train_model(cfg, model, train_loader, val_loader, class_weights, device, out_dir)

    peak_cpu_mb = max(compute_mod.peak_cpu_memory_bytes(), cpu0) / (1024 ** 2)
    model.load_state_dict(result["best_state"])

    val_metrics, val_pred = evaluate(model, val_loader, device)
    save_json(val_metrics, out_dir / "metrics.json")
    val_pred.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(result["history"]).to_csv(out_dir / "training_history.csv", index=False)

    ckpt_path = out_dir / "checkpoints" / "best_model.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": result["best_state"],
            "history": result["history"],
            "config": cfg,
            "val_metrics_summary": {
                k: val_metrics[k] for k in ("macro_f1", "qwk", "balanced_accuracy", "accuracy")
            },
            "epoch_selected_by": "val_macro_f1",
        },
        ckpt_path,
    )

    comp["training_time_s"] = round(result["training_time_s"], 1)
    comp["best_epoch"] = result["best_epoch"]
    comp["epochs_run"] = result["epochs_run"]
    comp["checkpoint_file_bytes"] = ckpt_path.stat().st_size
    comp["peak_cpu_rss_training_mb"] = round(peak_cpu_mb, 1)
    save_json(comp, out_dir / "computational_metrics.json")

    from src.tracking import save_run_metadata

    save_run_metadata(
        out_dir, cfg, freeze["dataset_version"], split_ver, result["epochs_run"],
        extra={
            "freeze_id": freeze["freeze_id"],
            "patient_disjointness_verified": True,
            "class_weights_train_only": [float(w) for w in class_weights],
            "selection_metric": "val_macro_f1",
            "val_metrics_summary": {k: val_metrics[k]
                                    for k in ("macro_f1", "qwk", "balanced_accuracy", "accuracy")},
            "training_time_s": result["training_time_s"],
        },
    )

    print("\n===== VAL RESULTS (test untouched) =====", flush=True)
    print(f"{exp_id} | acc={val_metrics['accuracy']:.4f} macroF1={val_metrics['macro_f1']:.4f} "
          f"bAcc={val_metrics['balanced_accuracy']:.4f} QWK={val_metrics['qwk']:.4f}", flush=True)
    print(f"out dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
