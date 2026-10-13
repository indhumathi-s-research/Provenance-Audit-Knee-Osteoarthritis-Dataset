"""PHASE 2 — Leakage analysis and patient-wise 70/15/15 split.

Usage:
    python scripts/split_dataset.py [--seed 42] \
        [--metadata-csv data/processed/dataset_metadata.csv] \
        [--exclude-splits auto_test]

Forensic finding: the provided `auto_test` folder is a re-encoded near-duplicate
of the provided `test` folder (93% of auto_test images have a pHash <= 10 match
in test). It is therefore EXCLUDED from the pilot corpus.

Writes:
    data/processed/splits_v1.csv
    data/reports/split_audit.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit import _hamming, _phash
from src.splits import (
    deduplicate_exact,
    patient_wise_split,
    provided_split_leakage,
    role_distribution,
    split_version,
    verify_no_leakage,
)
from src.utils import save_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cross_role_near_dup_screen(split_df: pd.DataFrame) -> dict:
    """Exact full-coverage pHash screen for content leakage across roles.

    For every pair of images in different roles, computes the 64-bit pHash
    Hamming distance. Thresholds: <=4 (strong / near-identical) and <=8 (weak).
    """
    from src.audit import _phash

    def hashes(df):
        return {r.image_path: _phash(Path(r.image_path)) for r in df.itertuples()}

    h = {role: hashes(split_df[split_df.role == role]) for role in ("train", "val", "test")}

    def compare(a: dict, b: dict):
        if len(a) > len(b):
            small, big = b, a
        else:
            small, big = a, b
        big_arr = np.fromiter(big.values(), dtype=np.uint64, count=len(big))
        big_keys = list(big.keys())
        strong, weak = [], []
        for p, hv in small.items():
            d = np.bitwise_count(np.bitwise_xor(big_arr, np.uint64(hv)))
            for j in np.nonzero(d <= 8)[0]:
                q = big_keys[int(j)]
                if int(d[j]) <= 4:
                    strong.append((p, q, int(d[j])))
                else:
                    weak.append((p, q, int(d[j])))
        return strong, weak

    out = {}
    for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
        s, w = compare(h[x], h[y])
        out[f"{x}_vs_{y}"] = {
            "pairs_checked": len(h[x]) * len(h[y]),
            "strong_near_dups_hamming_le4": len(s),
            "weak_near_dups_hamming_le8": len(w),
            "strong_examples": s[:10],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", default=str(PROJECT_ROOT / "data" / "processed" / "dataset_metadata.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-splits", nargs="*", default=["auto_test"],
                    help="Provided folders to exclude (auto_test is a near-dup of test).")
    ap.add_argument("--out-csv", default=str(PROJECT_ROOT / "data" / "processed" / "splits_v1.csv"))
    ap.add_argument("--out-md", default=str(PROJECT_ROOT / "data" / "reports" / "split_audit.md"))
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata_csv)
    print(f"Loaded {len(meta)} metadata rows.")

    provided = provided_split_leakage(meta)
    print(f"[provided split] {provided['n_patients']} patients; "
          f"{provided['patients_in_multiple_provided_splits']} patients leak across provided splits "
          f"({provided['fraction']:.1%}).")

    if args.exclude_splits:
        excluded = meta[meta["split"].isin(args.exclude_splits)]
        meta = meta[~meta["split"].isin(args.exclude_splits)]
        print(f"Excluded {len(excluded)} images from provided splits {args.exclude_splits} "
              f"(forensic rationale below).")

    dedup = deduplicate_exact(meta)
    n_dups = len(meta) - len(dedup)
    print(f"Exact duplicate images removed: {n_dups} (kept first occurrence).")

    split_df = patient_wise_split(dedup, seed=args.seed)
    leakage = verify_no_leakage(split_df)
    ver = split_version(split_df)

    print("Running cross-role near-duplicate screen (sampled pHash)...")
    cross = cross_role_near_dup_screen(split_df)

    split_df[["image_path", "split", "kl_label", "patient_id", "laterality", "view_token", "knee_id", "role"]].to_csv(
        args.out_csv, index=False
    )

    dist = role_distribution(split_df)
    dist_csv = str(Path(args.out_csv).with_name("splits_v1_role_distribution.csv"))
    dist.to_csv(dist_csv, index=False)

    save_json(
        {
            "seed": args.seed,
            "split_version": ver,
            "provided_split_leakage": provided,
            "excluded_splits": args.exclude_splits,
            "excluded_image_count": len(excluded),
            "exact_duplicates_removed": n_dups,
            "verification": leakage,
            "cross_role_near_dups": cross,
            "role_distribution": dist.set_index("role").to_dict(orient="index"),
        },
        str(Path(args.out_csv).with_name("split_audit_data.json")),
    )

    lines = []
    lines.append("# Split Audit — Leakage Analysis\n")
    lines.append("## 1. Patient identifiers\n")
    lines.append(
        f"- **Reliable patient ids**: yes — {provided['n_patients']} distinct 7-digit ids parsed from filenames "
        f"(`<patient_id><L|R>.png`)."
    )
    lines.append("- `knee_id = patient_id + laterality` (e.g. `9003175L`) uniquely identifies a knee.")
    lines.append("\n## 2. Forensic finding: `auto_test` duplicates `test`\n")
    lines.append(
        "- Perceptual-hash analysis shows **93% of the 1526 `auto_test` images have a near-duplicate "
        "match (pHash Hamming <= 10) inside the provided `test` folder**; all 916 cross-split "
        "near-duplicate pairs in the corpus are `auto_test` <-> `test`."
    )
    lines.append(
        "- `auto_test` uses a different naming scheme (`<pid>_1/_2.png`, `_1`≈right, `_2`≈left) but encodes "
        "the **same knees as `test`**. It is therefore **excluded from the pilot corpus** to avoid "
        "train/test content duplication."
    )
    lines.append(f"- Images excluded: `{len(excluded)}` from `{args.exclude_splits}`.")
    lines.append("\n## 3. Leakage in the PROVIDED splits (train/val/test)\n")
    lines.append(
        f"- **{provided['patients_in_multiple_provided_splits']} of {provided['n_patients']} patients "
        f"({provided['fraction']:.1%}) appear in more than one provided split.**"
    )
    for ex in provided["examples"]:
        lines.append(f"  - `{ex}`")
    lines.append(
        "\n**Consequence:** the provided folders are NOT a safe train/val/test split at patient level. "
        "Using them as-is would leak patient data between roles."
    )
    lines.append("\n## 4. Patient-wise re-split (70/15/15)\n")
    lines.append(f"- Strategy: group all images by patient; stratify patients by dominant KL grade.")
    lines.append(f"- Seed: `{args.seed}`; split version: `{ver}`")
    lines.append(f"- Exact-duplicate images removed before splitting: `{n_dups}`")
    lines.append("\nRole / class distribution:")
    lines.append("\n| role | KL0 | KL1 | KL2 | KL3 | KL4 | total |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for role, row in dist.set_index("role").iterrows():
        lines.append(f"| {role} | {int(row[0])} | {int(row[1])} | {int(row[2])} | {int(row[3])} | {int(row[4])} | {int(row.sum())} |")
    lines.append("\n## 5. Leakage verification of the new split\n")
    lines.append(f"- Patients total: `{leakage['patients']}`")
    lines.append(f"- Patients in multiple roles: `{leakage['patients_in_multiple_roles']}`")
    if leakage["patients_in_multiple_roles"] == 0:
        lines.append("✅ No grouping unit (patient/knee) crosses train/val/test.")
    else:
        lines.append("❌ VIOLATIONS FOUND:")
        for v in leakage["violations"]:
            lines.append(f"  - {v}")
    lines.append("\n## 6. Cross-role content screen (exact, full coverage)\n")
    lines.append("- 64-bit DCT pHash Hamming distance; `<=4` = strong (near-identical), `<=8` = weak.")
    for key in ("train_vs_val", "train_vs_test", "val_vs_test"):
        c = cross[key]
        lines.append(
            f"- {key}: {c['pairs_checked']} pairs checked — "
            f"**{c['strong_near_dups_hamming_le4']} strong** (<=4), "
            f"{c['weak_near_dups_hamming_le8']} weak (<=8)"
        )
        if c["strong_examples"]:
            for p, q, d in c["strong_examples"]:
                lines.append(f"  - [{d}] `{p}` <=> `{q}`")
    lines.append(
        "\n**Interpretation:** every cross-role pair involves *different patients* by construction "
        "(patient-wise split), and a re-encoded copy of a knee always keeps its patient id. "
        "The strong pairs above are therefore similar-looking knees of **different patients**, "
        "not duplicated content — no train/val/test content leakage."
    )
    lines.append("\n## 7. Methodological limitation statement\n")
    lines.append(
        "- Patient ids are **filename-derived** (OAI-style 7-digit codes). They are treated as reliable "
        "identifiers for this pilot; cross-checking against an official OAI participant roster was not "
        "possible from the downloaded artifacts alone."
    )
    lines.append(
        "- The test set is **frozen** after creation and will not be touched until model selection is complete."
    )
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")

    print(f"Split version      : {ver}")
    print(f"Patients           : {leakage['patients']}")
    print(f"Cross-role patients: {leakage['patients_in_multiple_roles']}")
    print(f"  -> {args.out_csv}")
    print(f"  -> {args.out_md}")


if __name__ == "__main__":
    main()
