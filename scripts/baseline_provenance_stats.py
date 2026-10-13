"""Reproducible provenance statistics for the two baseline generations (read-only on stored run files).

Writes research/results/baseline_provenance_stats.json with:
  - per-model validation/test metrics, epochs run and best epoch for generation A (runs/) and B (experiments/baseline/)
  - M6 generation A vs B: test-label agreement of the two trained models
  - patient-cluster bootstrap interval (seeded) of generation-B M6 test and validation macro-F1
    (single trained model; excludes training-seed variance)
No model is trained or evaluated; only stored predictions and metric files are read.
"""
from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "results" / "baseline_provenance_stats.json"
MODELS = ["M1", "M2", "M3", "M4", "M5", "M6"]
N_BOOT, SEED = 1000, 0


def a_dir(m: str) -> Path:
    return Path(glob.glob(str(ROOT / "runs" / f"{m}_seed42_*"))[0])


def boot_ci(df: pd.DataFrame) -> dict:
    pid = df["patient_id"].astype(str).to_numpy()
    y, p = df["true_label"].to_numpy(), df["predicted_label"].to_numpy()
    groups = {k: np.where(pid == k)[0] for k in np.unique(pid)}
    keys = list(groups)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        vals.append(f1_score(y[idx], p[idx], average="macro"))
    return {"macro_f1": round(float(f1_score(y, p, average="macro")), 6),
            "ci95_low": round(float(np.percentile(vals, 2.5)), 4),
            "ci95_high": round(float(np.percentile(vals, 97.5)), 4),
            "patients": len(keys), "images": int(len(df)), "n_boot": N_BOOT, "seed": SEED}


def head() -> dict:
    def run(*a):
        r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"head": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


res: dict = {"meta": {"script": "scripts/baseline_provenance_stats.py", "git": head()}, "per_model": {}}
for m in MODELS:
    a, b = a_dir(m), ROOT / "experiments" / "baseline" / m
    ha = pd.read_csv(a / "history.csv")
    hb = pd.read_csv(b / "training_history.csv")
    va = json.loads((a / "val_metrics.json").read_text())
    vb = json.loads((b / "metrics.json").read_text())
    ta = json.loads((a / "test_metrics.json").read_text())
    tb_path = b / "test_metrics.json"
    res["per_model"][m] = {
        "gen_A": {"val_macro_f1": round(va["macro_f1"], 4), "test_macro_f1": round(ta["macro_f1"], 4),
                  "test_qwk": round(ta["qwk"], 4), "epochs_run": int(len(ha)),
                  "best_epoch": int(ha.val_macro_f1.idxmax()) + 1, "test_evaluated": True},
        "gen_B": {"val_macro_f1": round(vb["macro_f1"], 4), "val_qwk": round(vb["qwk"], 4),
                  "epochs_run": int(len(hb)), "best_epoch": int(hb.val_macro_f1.idxmax()) + 1,
                  "test_evaluated": tb_path.exists(),
                  **({"test_macro_f1": round(json.loads(tb_path.read_text())["macro_f1"], 4),
                      "test_qwk": round(json.loads(tb_path.read_text())["qwk"], 4)} if tb_path.exists() else {})},
    }

pa = pd.read_csv(a_dir("M6") / "predictions_test.csv")
pb = pd.read_csv(ROOT / "experiments" / "baseline" / "M6" / "predictions_test.csv")
mg = pa.merge(pb, on="image_path", suffixes=("_A", "_B"))
assert len(mg) == len(pa) == len(pb) and (mg.true_label_A == mg.true_label_B).all()
res["M6_gen_A_vs_B_test"] = {
    "images": int(len(mg)),
    "predicted_label_agreement": round(float((mg.predicted_label_A == mg.predicted_label_B).mean()), 6),
    "macro_f1_A": round(float(f1_score(mg.true_label_A, mg.predicted_label_A, average="macro")), 6),
    "macro_f1_B": round(float(f1_score(mg.true_label_B, mg.predicted_label_B, average="macro")), 6),
}
res["M6_gen_B_bootstrap"] = {
    "test": boot_ci(pb),
    "validation": boot_ci(pd.read_csv(ROOT / "experiments" / "baseline" / "M6" / "predictions.csv")),
    "note": "single trained model; patient-cluster bootstrap; excludes training-seed variance",
}
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("M6_gen_A_vs_B_test", "M6_gen_B_bootstrap")}, indent=2))
