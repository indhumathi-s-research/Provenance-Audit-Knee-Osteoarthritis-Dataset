"""PHASE 8 — Build reports/pilot_baseline_comparison.md from completed runs.

Scans runs/ for the latest run directory of each experiment id (M1..M6),
loads validation/test metrics, compute profiles and run metadata, then writes
the pilot baseline comparison and the final A-H methodological report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_latest_runs(runs_dir: Path) -> dict:
    """Return {experiment_id: run_dir} choosing the latest run per id."""
    best = {}
    if not runs_dir.is_dir():
        return best
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        md = d / "run_metadata.json"
        if not md.exists():
            continue
        meta = load_json(md)
        exp_id = meta.get("experiment", {}).get("id", d.name)
        if exp_id not in best:
            best[exp_id] = d
    return best


def fmt(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_ms(d: dict) -> str:
    if not d:
        return "—"
    return f"{d['mean_ms']:.2f}±{d['std_ms']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "pilot_baseline_comparison.md"))
    args = ap.parse_args()

    runs = find_latest_runs(Path(args.runs_dir))
    if not runs:
        raise SystemExit("no completed runs found in runs/")

    order = ["M1", "M2", "M3", "M4", "M5", "M6"]
    order = [k for k in order if k in runs] + [k for k in runs if k not in order]
    rows = []
    for exp_id in order:
        d = runs[exp_id]
        meta = load_json(d / "run_metadata.json")
        cfg = load_json(d / "config.json")
        test = load_json(d / "test_metrics.json")
        val = load_json(d / "val_metrics.json")
        comp = load_json(d / "compute.json") if (d / "compute.json").exists() else None
        hist = []
        if (d / "history.csv").exists():
            import pandas as pd
            hist = pd.read_csv(d / "history.csv").to_dict(orient="records")
        best_epoch = max((h for h in hist), key=lambda h: h.get("val_macro_f1", -1)) if hist else None
        rows.append(
            {
                "id": exp_id,
                "name": cfg.get("experiment", {}).get("name", ""),
                "arch": cfg.get("model", {}).get("architecture", ""),
                "res": "x".join(map(str, cfg.get("preprocessing", {}).get("image_size", []))),
                "contrast": cfg.get("preprocessing", {}).get("contrast", "none"),
                "roi": bool(cfg.get("roi", {}).get("enabled", False)),
                "test": test,
                "val": val,
                "compute": comp,
                "meta": meta,
                "history": hist,
                "best_epoch": best_epoch,
                "run_dir": d,
            }
        )

    L = []
    L.append("# Pilot Baseline Comparison (M1–M6)\n")
    L.append("Single-view KL grading on the piloted knee-OA dataset (auto_test excluded; "
             "patient-wise 70/15/15 split). Weighted cross-entropy (train-only weights) is the "
             "single imbalance strategy for all runs. Fixed seed 42, identical val/test protocol.\n")

    L.append("## 1. Headline results (test set)\n")
    L.append("| Config | Res | Contrast | ROI | Acc | Macro-F1 | Bal-Acc | QWK | Best-epoch | Train s |")
    L.append("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        t, m = r["test"], r["meta"]
        be = r["best_epoch"]
        L.append(
            f"| {r['id']} ({r['name']}) | {r['res']} | {r['contrast']} | {r['roi']} | "
            f"{fmt(t['accuracy'])} | **{fmt(t['macro_f1'])}** | {fmt(t['balanced_accuracy'])} | "
            f"{fmt(t['qwk'])} | {be['epoch'] if be else '—'} | "
            f"{m.get('training_time_s','—') if isinstance(m.get('training_time_s'), float) else fmt(m.get('training_time_s'),1)} |"
        )

    L.append("\n## 2. Per-class metrics on test (for each config)\n")
    for r in rows:
        L.append(f"\n### {r['id']} — {r['name']}\n")
        L.append("| Class | Precision | Recall | F1 | AUC-ovr |")
        L.append("|:--|--:|--:|--:|--:|")
        for c in r["test"]["per_class"]:
            L.append(
                f"| {c['class']} | {fmt(c['precision'])} | {fmt(c['recall'])} | {fmt(c['f1'])} | "
                f"{fmt(c['auc_ovr'])} |"
            )

    L.append("\n## 3. Computational profile\n")
    L.append("Hardware: see per-run metadata. Precision: FP32. Batch-1 latency "
             "(50 reps, warm-up 10, CUDA-synced): mean±std.\n")
    L.append("| Config | Params | MACs | Size | Preprocess ms | Inf lat b1 ms | Throughput b1 | Throughput b32 | Peak mem inf | Peak mem train |")
    L.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        c = r["compute"]
        if not c:
            L.append(f"| {r['id']} | — | — | — | — | — | — | — | — | — |")
            continue
        lat = c.get("inference_latency_batch1_ms", {})
        pre = c.get("preprocessing_latency_ms", {})
        th1 = c.get("throughput_batch1", {})
        th32 = c.get("throughput_batch32", {})
        mi = c.get("peak_memory_inference_mb", {})
        mt = c.get("peak_memory_training_mb", {})
        size = c.get("model_size_bytes", 0) / (1024 ** 2)
        L.append(
            f"| {r['id']} | {c.get('parameters', 0):,} | {c.get('macs', 0):,} | {size:.2f} MB | "
            f"{fmt_ms(pre)} | {fmt_ms(lat)} | {th1.get('images_per_second', 0):.0f} | "
            f"{th32.get('images_per_second', 0):.0f} | "
            f"{fmt(mi.get('peak_allocated_mb'), 1) if mi.get('peak_allocated_mb') else '—'} | "
            f"{fmt(mt.get('peak_allocated_mb'), 1) if mt.get('peak_allocated_mb') else '—'} |"
        )

    L.append("\n## 4. Validation vs test (consistency check)\n")
    L.append("| Config | Val Macro-F1 | Test Macro-F1 | Val QWK | Test QWK |")
    L.append("|:--|--:|--:|--:|--:|")
    for r in rows:
        L.append(
            f"| {r['id']} | {fmt(r['val']['macro_f1'])} | {fmt(r['test']['macro_f1'])} | "
            f"{fmt(r['val']['qwk'])} | {fmt(r['test']['qwk'])} |"
        )

    # Rank by primary metric (macro-F1) and report strongest baseline.
    ranked = sorted(rows, key=lambda r: r["test"]["macro_f1"], reverse=True)
    best = ranked[0]
    L.append("\n## 5. Strongest pilot baseline\n")
    L.append(
        f"- **{best['id']} ({best['name']})** ranks first on test **macro-F1 "
        f"({best['test']['macro_f1']:.4f})**, QWK {best['test']['qwk']:.4f}, "
        f"balanced accuracy {best['test']['balanced_accuracy']:.4f}."
    )
    L.append("- This is a *pilot* ranking with a single seed; statistical significance "
             "is not claimed without repeated runs.")

    L.append("\n## 6. Accuracy–latency trade-off\n")
    for r in ranked:
        c = r["compute"]
        lat = c["inference_latency_batch1_ms"]["mean_ms"] if c else None
        L.append(
            f"- {r['id']}: test macro-F1 {r['test']['macro_f1']:.4f}, "
            f"QWK {r['test']['qwk']:.4f}, batch-1 latency {fmt(lat, 2)} ms, "
            f"params {c['parameters']:,}."
        )
    L.append("\nNo configuration is called 'real-time' — deployment latency claims require "
             "the full inference stack and are out of pilot scope.\n")

    L.append("\n## 7. Methodological limitations (for the paper)\n")
    L.append(
        "1. **Single-view dataset.** No PA/AP-vs-LAT information exists in filenames or content; "
        "L/R is laterality, not view. Multi-view claims require a genuinely paired-view dataset "
        "(future phase)."
    )
    L.append(
        "2. **auto_test exclusion.** The provided `auto_test` folder was excluded because 93% of "
        "its images are perceptual duplicates of the provided `test` folder; excluding avoids "
        "train/test content duplication."
    )
    L.append(
        "3. **Patient-identifier provenance.** Patient ids are filename-derived (OAI-style). They "
        "were not cross-checked against the official OAI roster in this pilot."
    )
    L.append(
        "4. **Provided splits leaked.** 19.6% of patients appear in multiple provided splits; "
        "the pilot therefore uses a fresh patient-wise 70/15/15 split (split version "
        "`499379538d75`), and the test set is frozen."
    )
    L.append(
        "5. **Class imbalance.** Only weighted cross-entropy (train-only weights) is used; "
        "oversampling/focal loss were not combined."
    )
    L.append(
        "6. **Single seed, single split.** Pilot results are single-seed point estimates; "
        "the final protocol must add repeated seeds and statistical comparison."
    )
    L.append(
        "7. **Compute environment.** GTX 1050 Ti (4 GB), FP32; measurements are batch-1 with "
        "warm-up and CUDA sync; they are not 'real-time' claims."
    )
    L.append(
        "8. **Pretrained weights (M6).** M6 initializes from ImageNet weights (external "
        "distribution); other configs train from scratch, so M6 vs M1–M5 is not a controlled "
        "architecture-only comparison."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
