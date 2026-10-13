"""PHASE 1 — Dataset forensic audit.

Usage:
    python scripts/inspect_dataset.py --data-dir data

Writes:
    data/reports/dataset_audit.md
    data/reports/dataset_statistics.csv
    data/processed/dataset_metadata.csv
    data/processed/dataset_version.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit import run_audit, write_report
from src.metadata import build_metadata
from src.utils import save_json, sha256_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Root containing split/class folders")
    ap.add_argument("--out-md", default=str(PROJECT_ROOT / "data" / "reports" / "dataset_audit.md"))
    ap.add_argument("--out-csv", default=str(PROJECT_ROOT / "data" / "reports" / "dataset_statistics.csv"))
    args = ap.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.is_dir():
        raise SystemExit(f"data dir not found: {data_root}")

    print(f"Running forensic audit on {data_root} ...")
    summary, meta = run_audit(data_root)
    write_report(summary, args.out_md, args.out_csv)

    processed = PROJECT_ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    meta.to_csv(processed / "dataset_metadata.csv", index=False)

    version = sha256_text(
        ",".join(
            meta.sort_values("image_path")
            .apply(lambda r: f"{r['image_path']}|{r['split']}|{r['kl_label']}", axis=1)
            .tolist()
        )
    )[:16]
    (processed / "dataset_version.txt").write_text(version, encoding="utf-8")
    save_json(summary, processed / "audit_summary.json")

    print("Audit complete.")
    print(f"  total images       : {summary['total_images']}")
    print(f"  patients parsed    : {summary.get('patient_stats', {}).get('n_patients')}")
    print(f"  exact dup groups   : {len(summary['duplicates']['exact_groups'])}")
    print(f"  corrupt files      : {summary['verify']['header_failed']}")
    print(f"  dataset version    : {version}")
    print(f"  -> {args.out_md}")
    print(f"  -> {args.out_csv}")


if __name__ == "__main__":
    main()
