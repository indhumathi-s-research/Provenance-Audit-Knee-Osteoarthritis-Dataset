"""FREEZE the audited pilot corpus as KOA_KL_PILOT_v1.

Usage:
    python scripts/freeze_dataset.py

Records (read-only; no image is modified):
    - dataset version hash + split version hash
    - role sizes and KL distributions
    - patient-disjointness verification of the frozen split
    - exclusions (auto_test)
Writes:
    data/processed/KOA_KL_PILOT_v1_freeze.json
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.splits import verify_no_leakage
from src.utils import save_json, sha256_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREEZE_ID = "KOA_KL_PILOT_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(PROJECT_ROOT / "data" / "processed" / "splits_v1.csv"))
    ap.add_argument("--dataset-version-file",
                    default=str(PROJECT_ROOT / "data" / "processed" / "dataset_version.txt"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed" / f"{FREEZE_ID}_freeze.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.splits)
    allowed = {"train", "val", "test"}
    bad = set(df["role"].unique()) - allowed
    if bad:
        raise SystemExit(f"unexpected roles in splits file: {bad}")
    if df["split"].eq("auto_test").any():
        raise SystemExit("auto_test images found in splits file — must be excluded")

    leak = verify_no_leakage(df)
    if leak["patients_in_multiple_roles"] > 0:
        raise SystemExit(f"patient leakage detected: {leak['violations'][:3]}")

    dist = (
        df.groupby(["role", "kl_label"]).size().unstack(fill_value=0).reindex(sorted(allowed))
    )
    dataset_version = Path(args.dataset_version_file).read_text(encoding="utf-8").strip()
    split_ver = sha256_text(
        ",".join(
            df.sort_values("image_path")
            .apply(lambda r: f"{r['image_path']}|{r['role']}", axis=1)
            .tolist()
        )
    )[:12]

    freeze = {
        "freeze_id": FREEZE_ID,
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_version": dataset_version,
        "split_version": split_ver,
        "images_total": int(len(df)),
        "patients_total": int(df["patient_id"].nunique()),
        "knees_total": int(df["knee_id"].nunique()),
        "roles": {r: int(n) for r, n in df.groupby("role").size().items()},
        "kl_distribution_by_role": {str(k): {int(c): int(v) for c, v in row.items()}
                                    for k, row in dist.iterrows()},
        "patient_disjointness_verified": True,
        "patients_in_multiple_roles": leak["patients_in_multiple_roles"],
        "excluded": ["auto_test/ (perceptual duplicates of provided test/)"],
        "label_source": "parent class folder 0-4 (Kaggle Knee Osteoarthritis Severity Grading)",
        "multi_view_available": False,
        "rules": [
            "original images are immutable",
            "frozen split must not be regenerated",
            "test set reserved for final evaluation only",
        ],
    }
    save_json(freeze, Path(args.out))
    print(f"FROZEN {FREEZE_ID} -> {args.out}")
    print(f"  images={freeze['images_total']} patients={freeze['patients_total']} "
          f"knees={freeze['knees_total']} roles={freeze['roles']}")


if __name__ == "__main__":
    main()
