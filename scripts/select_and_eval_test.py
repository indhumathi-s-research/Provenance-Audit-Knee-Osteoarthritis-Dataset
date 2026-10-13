"""BASELINE STAGE — select best config on VALIDATION, evaluate it on TEST once.

Selection criterion: highest validation macro-F1 (tie-break: higher QWK).
Only the selected configuration is ever evaluated on the frozen test set.

Usage:
    python scripts/select_and_eval_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_paths
from src.models import build_model
from src.splits import verify_no_leakage
from src.train_eval import build_loaders, evaluate
from src.utils import get_device, load_json, save_json, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "experiments" / "baseline"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits_v1.csv"


def main():
    rows = []
    for d in sorted(BASE.iterdir()):
        mpath = d / "metrics.json"
        if not d.is_dir() or not mpath.exists():
            continue
        m = load_json(mpath)
        rows.append({"id": d.name, "dir": d, "macro_f1": m["macro_f1"], "qwk": m["qwk"]})
    if not rows:
        raise SystemExit("no completed baseline runs found")

    ranked = sorted(rows, key=lambda r: (r["macro_f1"], r["qwk"]), reverse=True)
    selected = ranked[0]
    print("Validation ranking (selection uses ONLY this):")
    for r in ranked:
        print(f"  {r['id']}: macroF1={r['macro_f1']:.4f} QWK={r['qwk']:.4f}"
              + ("   <-- SELECTED" if r is selected else ""))

    save_json(
        {
            "selected_id": selected["id"],
            "criterion": "max validation macro_f1 (tie-break qwk)",
            "ranking": [{"id": r["id"], "val_macro_f1": r["macro_f1"], "val_qwk": r["qwk"]}
                        for r in ranked],
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        BASE / "selection.json",
    )

    cfg_path = selected["dir"] / "config.yaml"
    import yaml

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    class _C(dict):
        pass

    cfg = load_config(cfg_path)
    resolve_paths(cfg, str(PROJECT_ROOT))
    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    device = get_device()

    split_df = pd.read_csv(SPLITS_PATH)
    leak = verify_no_leakage(split_df)
    if leak["patients_in_multiple_roles"] > 0:
        raise SystemExit("ABORT: split not patient-disjoint")

    _, _, test_loader, _, _ = build_loaders(cfg, split_df, device, seed)
    model = build_model(cfg).to(device)
    ckpt = torch.load(selected["dir"] / "checkpoints" / "best_model.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"\nEvaluating SELECTED config {selected['id']} on the frozen TEST set ...")
    test_metrics, test_pred = evaluate(model, test_loader, device)
    save_json(test_metrics, selected["dir"] / "test_metrics.json")
    test_pred.to_csv(selected["dir"] / "predictions_test.csv", index=False)

    print("\n===== FINAL HELD-OUT TEST RESULTS =====")
    for k in ("accuracy", "macro_f1", "balanced_accuracy", "qwk"):
        print(f"  {k}: {test_metrics[k]:.4f}")
    for c in test_metrics["per_class"]:
        print(f"  {c['class']}: P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}")


if __name__ == "__main__":
    main()
