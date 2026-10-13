"""OPTIMIZATION STAGE — hyperparameter search over M6's space.

Three methods (Random Search, PSO, SOA) share the SAME search space, training
budget, validation protocol and evaluation budget (README: "must share the same
search space, evaluation budget, validation protocol and training budget").

Each trial:
    - decodes a box point -> config overrides on top of M6 (ResNet-18 pretrained
      + CLAHE + ImageNet norm)
    - trains with weighted CE (train-only class weights) under the frozen,
      patient-disjoint split
    - reports VALIDATION macro-F1 as the objective (test is NEVER loaded here)

The test set is touched exactly ONCE, later, by the global best config only
(see scripts/optimize_report.py).

Outputs per method under experiments/optimization/<method>/:
    results.csv          (every objective query: human params + metrics + cached?)
    eval_cache.json      (vec_key -> val_macro_f1, for resume without retraining)
    state.json           (PSO/SOA population state + RNG state, for resume)
    trial_<N>/           (config.yaml, metrics.json, predictions.csv,
                          checkpoints/best_model.pt, training_history.csv,
                          run_metadata.json)  -- only for UNIQUE trainings

Usage:
    python scripts/optimize.py --method random --trials 20
    python scripts/optimize.py --method pso    --trials 20
    python scripts/optimize.py --method soa    --trials 20
    python scripts/optimize.py --method all    --trials 20   # sequential, background-safe
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import deep_merge, load_config, resolve_paths
from src.dataset import KneeImageDataset, class_weights_from_df
from src.models import build_model
from src.splits import verify_no_leakage
from src.train_eval import evaluate, train_model
from src.utils import get_device, save_json, set_seed, sha256_text
from src.search_space import DIM, LB, UB, decode, human_readable, random_vector, vector_key

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREEZE_PATH = PROJECT_ROOT / "data" / "processed" / "KOA_KL_PILOT_v1_freeze.json"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits_v1.csv"
M6_YAML = PROJECT_ROOT / "configs" / "m6.yaml"
OUT_ROOT = PROJECT_ROOT / "experiments" / "optimization"
SEED = 42
METHOD_SEED_OFFSET = {"random": 0, "pso": 1000, "soa": 2000}


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
class Objective:
    """Trains a decoded config; caches results by (quantized) vector."""

    def __init__(self, method: str, smoke: bool = False):
        self.method = method
        self.smoke = smoke
        self.mdir = OUT_ROOT / method
        self.mdir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.mdir / "eval_cache.json"
        self.csv_path = self.mdir / "results.csv"
        self.cache: dict = {}
        self.unique = 0
        self.trial_counter = 0
        self._load_cache()
        self.split_df = pd.read_csv(SPLITS_PATH)
        self.freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        self.split_ver = sha256_text(
            ",".join(
                self.split_df.sort_values("image_path")
                .apply(lambda r: f"{r['image_path']}|{r['role']}", axis=1)
                .tolist()
            )
        )[:12]
        if self.split_ver != self.freeze["split_version"]:
            raise SystemExit(f"ABORT: split version mismatch {self.split_ver}")
        leak = verify_no_leakage(self.split_df)
        if leak["patients_in_multiple_roles"] > 0:
            raise SystemExit("ABORT: split not patient-disjoint")
        log(f"[{method}] objective ready | freeze={self.freeze['freeze_id']} "
            f"split_ok | cache={self.unique} prior unique trainings")

    def _load_cache(self):
        if self.cache_path.exists():
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.cache = {tuple(float(t) for t in k.split("|")): v for k, v in raw.items()}
            self.unique = len(self.cache)

    def _save_cache(self):
        payload = {"|".join(f"{x:.6f}" for x in k): v for k, v in self.cache.items()}
        self.cache_path.write_text(json.dumps(payload), encoding="utf-8")

    def __call__(self, vec: np.ndarray, tag: str = ""):
        key = vector_key(vec)
        if key in self.cache:
            val = self.cache[key]
            self._append_row(vec, val, cached=True, wall_s=0.0)
            return val
        # ---- unique training run ----
        self.trial_counter += 1
        t0 = time.perf_counter()
        val = self._train(vec, self.trial_counter)
        wall = time.perf_counter() - t0
        self.cache[key] = val
        self.unique += 1
        self._save_cache()
        self._append_row(vec, val, cached=False, wall_s=wall)
        log(f"[{self.method}] trial {self.trial_counter} (unique {self.unique}) "
            f"val_macroF1={val:.4f} {tag} [{wall/60:.1f} min]")
        return val

    def _append_row(self, vec, val, cached, wall_s):
        hr = human_readable(vec)
        row = {
            "trial": self.trial_counter, "method": self.method, "cached": cached,
            "val_macro_f1": round(val, 6), "wall_s": round(wall_s, 1),
            "timestamp": datetime.datetime.now().isoformat(),
            "vec": json.dumps([round(float(x), 6) for x in vec]),
            **hr,
        }
        df = pd.DataFrame([row])
        if self.csv_path.exists():
            df.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.csv_path, index=False)

    def _train(self, vec, trial_idx):
        overrides = decode(vec)
        cfg = load_config(str(M6_YAML))
        cfg = deep_merge(cfg, overrides)
        if self.smoke:
            cfg["training"]["epochs"] = 2
            cfg["training"]["max_train_batches"] = 20
        resolve_paths(cfg, str(PROJECT_ROOT))
        cfg["experiment"]["id"] = f"{self.method}_t{trial_idx}"
        cfg["experiment"]["seed"] = SEED
        trial_dir = self.mdir / f"trial_{trial_idx}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(trial_dir / "cache")

        set_seed(SEED)
        device = get_device()
        train_df = self.split_df[self.split_df.role == "train"]
        val_df = self.split_df[self.split_df.role == "val"]
        test_df = self.split_df[self.split_df.role == cfg["eval"]["test_role"]]
        from torch.utils.data import DataLoader

        train_ds = KneeImageDataset(train_df, cfg, train=True, cache_dir=cache_dir)
        val_ds = KneeImageDataset(val_df, cfg, train=False, cache_dir=cache_dir)
        test_ds = KneeImageDataset(test_df, cfg, train=False, cache_dir=None)  # untouched
        g = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                                 shuffle=True, num_workers=cfg["training"]["workers"],
                                 generator=g)
        val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                               shuffle=False, num_workers=cfg["training"]["workers"])
        _ = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                      shuffle=False, num_workers=cfg["training"]["workers"])
        weights = class_weights_from_df(train_df, cfg["model"]["num_classes"])

        model = build_model(cfg).to(device)
        result = train_model(cfg, model, train_loader, val_loader, weights, device, trial_dir)
        model.load_state_dict(result["best_state"])
        val_metrics, val_pred = evaluate(model, val_loader, device)

        with open(trial_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        save_json(val_metrics, trial_dir / "metrics.json")
        val_pred.to_csv(trial_dir / "predictions.csv", index=False)
        pd.DataFrame(result["history"]).to_csv(trial_dir / "training_history.csv", index=False)
        ckpt = trial_dir / "checkpoints" / "best_model.pt"
        ckpt.parent.mkdir(exist_ok=True)
        torch.save({"model_state_dict": result["best_state"], "config": cfg,
                    "val_metrics_summary": {k: val_metrics[k] for k in
                                            ("macro_f1", "qwk", "balanced_accuracy", "accuracy")},
                    "epoch_selected_by": "val_macro_f1"}, ckpt)
        save_json({
            "method": self.method, "trial": trial_idx, "seed": SEED,
            "freeze_id": self.freeze["freeze_id"], "split_version": self.split_ver,
            "hyperparameters": human_readable(vec),
            "val_macro_f1": val_metrics["macro_f1"], "val_qwk": val_metrics["qwk"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_selected": result["best_epoch"], "epochs_run": result["epochs_run"],
            "training_time_s": round(result["training_time_s"], 1),
            "class_weights_train_only": [float(w) for w in weights],
        }, trial_dir / "run_metadata.json")
        # free the per-trial image cache (only this trial's ~3GB)
        shutil.rmtree(cache_dir, ignore_errors=True)
        return float(val_metrics["macro_f1"])


# --------------------------------------------------------------------------- #
# Optimizers (operate in the [0,1]^D box)
# --------------------------------------------------------------------------- #
def _persist_state(method, state):
    (OUT_ROOT / method / "state.json").write_text(
        json.dumps(state, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o),
        encoding="utf-8")


def _load_state(method):
    p = OUT_ROOT / method / "state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def run_random(obj: Objective, budget: int, rng: np.random.Generator):
    while obj.unique < budget:
        obj(random_vector(rng))


def run_pso(obj: Objective, budget: int, rng: np.random.Generator):
    S, w, c1, c2 = 5, 0.7, 1.5, 1.5
    max_iter = budget  # enough iterations to reach `budget` unique trainings despite dedup
    st = _load_state("pso")
    if st and st.get("algo") == "pso":
        pos = np.array(st["pos"]); vel = np.array(st["vel"])
        pbest_pos = np.array(st["pbest_pos"]); pbest_val = np.array(st["pbest_val"])
        gbest_pos = np.array(st["gbest_pos"]); gbest_val = st["gbest_val"]
        it = st["iter"]
    else:
        pos = rng.random((S, DIM)); vel = (rng.random((S, DIM)) * 2 - 1) * 0.1
        pbest_pos = pos.copy(); pbest_val = np.full(S, -np.inf)
        gbest_pos = pos[0].copy(); gbest_val = -np.inf; it = 0
    while obj.unique < budget and it < max_iter:
        vals = np.array([obj(pos[i], f"pso#{it}.{i}") for i in range(S)])
        imp = vals > pbest_val
        pbest_pos[imp] = pos[imp]; pbest_val[imp] = vals[imp]
        gi = int(np.argmax(vals))
        if vals[gi] > gbest_val:
            gbest_val = vals[gi]; gbest_pos = pos[gi].copy()
        r1 = rng.random((S, DIM)); r2 = rng.random((S, DIM))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        vel = np.clip(vel, -0.3, 0.3)
        pos = np.clip(pos + vel, LB, UB)
        it += 1
        _persist_state("pso", {"algo": "pso", "pos": pos, "vel": vel,
                               "pbest_pos": pbest_pos, "pbest_val": pbest_val,
                               "gbest_pos": gbest_pos, "gbest_val": gbest_val,
                               "iter": it, "rng": rng.bit_generator.state})
    log(f"[pso] done | unique trainings={obj.unique} gbest_val={gbest_val:.4f}")


def run_soa(obj: Objective, budget: int, rng: np.random.Generator):
    N = 5
    max_iter = budget  # enough iterations to reach `budget` unique trainings despite dedup
    st = _load_state("soa")
    if st and st.get("algo") == "soa":
        pos = np.array(st["pos"]); gbest_pos = np.array(st["gbest_pos"])
        gbest_val = st["gbest_val"]; it = st["iter"]
    else:
        pos = rng.random((N, DIM)); gbest_pos = pos[0].copy()
        gbest_val = -np.inf; it = 0
    while obj.unique < budget and it < max_iter:
        vals = np.array([obj(pos[i], f"soa#{it}.{i}") for i in range(N)])
        gi = int(np.argmax(vals))
        if vals[gi] > gbest_val:
            gbest_val = vals[gi]; gbest_pos = pos[gi].copy()
        t = it + 1
        A = 2 - t * (2 / max_iter)
        B = 1 - t * (1 / max_iter)
        new = np.empty_like(pos)
        for i in range(N):
            r3 = rng.random(DIM); r4 = rng.random(DIM)
            ds = np.abs(B * (gbest_pos - pos[i]))
            M = A * np.exp(B * r3) * np.cos(2 * np.pi * r4)
            new[i] = ds * M + gbest_pos
        pos = np.clip(new, LB, UB)
        it += 1
        _persist_state("soa", {"algo": "soa", "pos": pos, "gbest_pos": gbest_pos,
                               "gbest_val": gbest_val, "iter": it,
                               "rng": rng.bit_generator.state})
    log(f"[soa] done | unique trainings={obj.unique} gbest_val={gbest_val:.4f}")


RUNNERS = {"random": run_random, "pso": run_pso, "soa": run_soa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["random", "pso", "soa", "all"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 20 batches")
    args = ap.parse_args()

    methods = ["random", "pso", "soa"] if args.method == "all" else [args.method]
    for method in methods:
        obj = Objective(method, smoke=args.smoke)
        rng = np.random.default_rng(SEED + METHOD_SEED_OFFSET[method])
        log(f"=== method={method} target_unique={args.trials} (already {obj.unique}) ===")
        RUNNERS[method](obj, args.trials, rng)
        log(f"=== method={method} finished: {obj.unique} unique trainings ===")


if __name__ == "__main__":
    main()
