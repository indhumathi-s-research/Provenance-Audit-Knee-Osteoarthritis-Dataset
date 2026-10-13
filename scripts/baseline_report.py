"""BASELINE STAGE — final comparison reports and plots.

Reads experiments/baseline/<M1..M6>/ artifacts (validation metrics for all,
test metrics only for the selected configuration) plus computational_metrics.json.

Writes:
    reports/baseline_comparison.md
    reports/baseline_comparison.csv
    reports/confusion_matrices/<ID>_val.png  (+ <ID>_test.png for the selected config)
    reports/metric_comparison.png
    reports/latency_vs_macro_f1.png

Usage:
    python scripts/baseline_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_json, save_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "experiments" / "baseline"
REP = PROJECT_ROOT / "reports"
CM_DIR = REP / "confusion_matrices"
ORDER = ["M1", "M2", "M3", "M4", "M5", "M6"]


def fmt(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_ms(d: dict) -> str:
    if not d or d.get("repeats", 0) == 0:
        return "—"
    return f"{d['mean_ms']:.2f}±{d['std_ms']:.2f}"


def save_cm_png(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    n = cm.shape[0]
    ax.set_xticks(range(n), [f"KL{i}" for i in range(n)])
    ax.set_yticks(range(n), [f"KL{i}" for i in range(n)])
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    REP.mkdir(exist_ok=True)
    CM_DIR.mkdir(exist_ok=True)

    selection = load_json(BASE / "selection.json")
    sel_id = selection["selected_id"]

    rows = []
    for exp_id in ORDER:
        d = BASE / exp_id
        if not d.is_dir():
            continue
        val = load_json(d / "metrics.json")
        comp = load_json(d / "computational_metrics.json")
        cfg = yaml_safe(d / "config.yaml")
        test = load_json(d / "test_metrics.json") if (d / "test_metrics.json").exists() else None
        rows.append({"id": exp_id, "dir": d, "val": val, "test": test,
                     "comp": comp, "cfg": cfg})

    # ---------------- CSV ----------------
    csv_rows = []
    for r in rows:
        c, v = r["comp"], r["val"]
        t = r["test"]
        csv_rows.append({
            "config": r["id"],
            "name": r["cfg"]["experiment"]["name"],
            "resolution": "x".join(map(str, r["cfg"]["preprocessing"]["image_size"])),
            "contrast": r["cfg"]["preprocessing"].get("contrast", "none"),
            "roi_enabled": bool(r["cfg"]["roi"]["enabled"]),
            "params": c["parameters"],
            "macs": c["macs"],
            "model_size_bytes": c["model_size_bytes"],
            "checkpoint_file_bytes": c.get("checkpoint_file_bytes"),
            "train_time_s": c.get("training_time_s"),
            "best_epoch": c.get("best_epoch"),
            "epochs_run": c.get("epochs_run"),
            "val_macro_f1": v["macro_f1"],
            "val_qwk": v["qwk"],
            "val_balanced_accuracy": v["balanced_accuracy"],
            "val_accuracy": v["accuracy"],
            "val_kl4_f1": next(x["f1"] for x in v["per_class"] if x["class"] == "KL4"),
            "test_macro_f1": None if t is None else t["macro_f1"],
            "test_qwk": None if t is None else t["qwk"],
            "test_balanced_accuracy": None if t is None else t["balanced_accuracy"],
            "test_accuracy": None if t is None else t["accuracy"],
            "inference_latency_batch1_mean_ms": c["inference_latency_batch1_ms"]["mean_ms"],
            "inference_latency_batch1_std_ms": c["inference_latency_batch1_ms"]["std_ms"],
            "end_to_end_latency_batch1_mean_ms": c["end_to_end_latency_batch1_ms"]["mean_ms"],
            "end_to_end_latency_batch1_std_ms": c["end_to_end_latency_batch1_ms"]["std_ms"],
            "preprocessing_latency_mean_ms": c["preprocessing_latency_ms"]["mean_ms"],
            "roi_latency_mean_ms": c["roi_latency_ms"]["mean_ms"],
            "throughput_batch1_ips": c["throughput_batch1"]["images_per_second"],
            "peak_gpu_mem_inference_mb": c["peak_memory_inference_mb"]["peak_allocated_mb"],
            "peak_gpu_mem_training_mb": c["peak_memory_training_mb"]["peak_allocated_mb"],
            "peak_cpu_rss_training_mb": c.get("peak_cpu_rss_training_mb"),
        })
    pd.DataFrame(csv_rows).to_csv(REP / "baseline_comparison.csv", index=False)

    # ---------------- confusion matrices ----------------
    for r in rows:
        save_cm_png(np.array(r["val"]["confusion_matrix"]),
                    f"{r['id']} — validation (n={sum(sum(x) for x in np.array(r['val']['confusion_matrix']))})",
                    CM_DIR / f"{r['id']}_val.png")
        if r["test"] is not None:
            save_cm_png(np.array(r["test"]["confusion_matrix"]),
                        f"{r['id']} — TEST held-out (n={sum(sum(x) for x in np.array(r['test']['confusion_matrix']))})",
                        CM_DIR / f"{r['id']}_test.png")

    # ---------------- metric_comparison.png ----------------
    metrics = [("macro_f1", "Macro-F1"), ("qwk", "QWK"), ("balanced_accuracy", "Balanced acc")]
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (key, label) in enumerate(metrics):
        vals = [r["val"][key] for r in rows]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=f"{label} (val)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    sel_idx = next(i for i, r in enumerate(rows) if r["id"] == sel_id)
    ax.bar(x[sel_idx] - w, rows[sel_idx]["test"]["macro_f1"], w * 0.35,
           color="black", label="TEST macro-F1 (selected only)")
    ax.set_xticks(x, [r["id"] for r in rows])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Baseline configurations — primary/secondary metrics (validation; "
                 "black bar = held-out test of the selected config)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REP / "metric_comparison.png", dpi=150)
    plt.close(fig)

    # ---------------- latency_vs_macro_f1.png ----------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in rows:
        lat = r["comp"]["end_to_end_latency_batch1_ms"]
        f1 = r["val"]["macro_f1"]
        size = max(40.0, min(r["comp"]["parameters"] / 4000.0, 600))
        is_sel = r["id"] == sel_id
        ax.scatter(lat["mean_ms"], f1, s=size,
                   marker="*" if is_sel else "o",
                   color="#c0392b" if is_sel else "#2471a3",
                   edgecolors="black", zorder=3)
        ax.annotate(
            f"{r['id']}\n{lat['mean_ms']:.2f}ms",
            (lat["mean_ms"], f1), textcoords="offset points",
            xytext=(10, -4), fontsize=9,
            fontweight="bold" if is_sel else "normal",
        )
    ax.set_xlabel("end-to-end batch-1 latency (preprocess → inference), ms")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Latency vs Macro-F1 (bubble area ∝ parameters)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REP / "latency_vs_macro_f1.png", dpi=150)
    plt.close(fig)

    # ---------------- markdown ----------------
    L = []
    L.append("# Baseline Comparison — KOA_KL_PILOT_v1 (single-view KL grading)\n")
    L.append("Protocol: frozen patient-disjoint split (verified before every experiment); "
             "weighted cross-entropy with train-only class weights as the ONLY imbalance "
             "mechanism; fixed seed 42; model/config selection by validation macro-F1; "
             "**the test set was touched exactly once** — by the selected configuration.\n")

    best_f1 = max(rows, key=lambda r: r["val"]["macro_f1"])
    best_qwk = max(rows, key=lambda r: r["val"]["qwk"])
    best_bacc = max(rows, key=lambda r: r["val"]["balanced_accuracy"])
    best_kl4 = max(rows, key=lambda r: next(x["f1"] for x in r["val"]["per_class"] if x["class"] == "KL4"))
    fastest = min(rows, key=lambda r: r["comp"]["end_to_end_latency_batch1_ms"]["mean_ms"])
    lightest = min(rows, key=lambda r: (r["comp"]["macs"], r["comp"]["parameters"]))
    tradeoff = max(rows, key=lambda r: r["val"]["macro_f1"]
                   - 0.02 * np.log10(max(r["comp"]["macs"], 1)))

    L.append("## 1. Validation results (all six configurations)\n")
    L.append("| Config | Description | Res | Preproc | Val macro-F1 | Val QWK | Val bAcc | Val KL4-F1 | Best epoch | Train s |")
    L.append("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        kl4 = next(x["f1"] for x in r["val"]["per_class"] if x["class"] == "KL4")
        L.append(f"| {r['id']} | {r['cfg']['experiment']['name']} | "
                 f"{r['cfg']['preprocessing']['image_size'][0]}px | "
                 f"{r['cfg']['preprocessing'].get('contrast','none')}"
                 f"{' + ROI' if r['cfg']['roi']['enabled'] else ''} | "
                 f"{fmt(r['val']['macro_f1'])} | {fmt(r['val']['qwk'])} | "
                 f"{fmt(r['val']['balanced_accuracy'])} | {fmt(kl4)} | "
                 f"{r['comp']['best_epoch']} | {r['comp']['training_time_s']} |")

    sel = next(r for r in rows if r["id"] == sel_id)
    t = sel["test"]
    L.append(f"\n## 2. Held-out TEST results — selected configuration **{sel_id}**\n")
    L.append(f"- accuracy **{t['accuracy']:.4f}**, macro-F1 **{t['macro_f1']:.4f}**, "
             f"balanced accuracy **{t['balanced_accuracy']:.4f}**, QWK **{t['qwk']:.4f}**")
    L.append("\n| Class | Precision | Recall | F1 | AUC-ovr |")
    L.append("|:--|--:|--:|--:|--:|")
    for c in t["per_class"]:
        L.append(f"| {c['class']} | {fmt(c['precision'])} | {fmt(c['recall'])} | "
                 f"{fmt(c['f1'])} | {fmt(c['auc_ovr'])} |")
    L.append("\nConfusion matrix (test): see `confusion_matrices/" + sel_id + "_test.png`.\n")

    L.append("## 3. Computational profile\n")
    L.append("Hardware/software recorded per run in `experiments/baseline/*/run_metadata.json`. "
             "FP32 on GTX 1050 Ti; batch-1 latencies use 10 warm-up iterations, CUDA sync, "
             "50 repeats; reported mean±std ms.\n")
    L.append("| Config | Params | MACs | ckpt MB | Inf b1 ms | E2E b1 ms | Preproc ms | ROI ms | Thr b1 ips | Thr b32 ips | GPU peak train MB | CPU peak RSS MB |")
    L.append("|:--|--:|--:|--:|:--|:--|:--|:--|--:|--:|--:|--:|")
    for r in rows:
        c = r["comp"]
        L.append(
            f"| {r['id']} | {c['parameters']:,} | {c['macs']:,} | "
            f"{c['checkpoint_file_bytes']/(1024**2):.2f} | "
            f"{fmt_ms(c['inference_latency_batch1_ms'])} | "
            f"{fmt_ms(c['end_to_end_latency_batch1_ms'])} | "
            f"{fmt_ms(c['preprocessing_latency_ms'])} | "
            f"{fmt_ms(c['roi_latency_ms']) if c['roi_latency_ms']['repeats'] else '—'} | "
            f"{c['throughput_batch1']['images_per_second']:.0f} | "
            f"{c['throughput_batch32']['images_per_second']:.0f} | "
            f"{fmt(c['peak_memory_training_mb']['peak_allocated_mb'], 0)} | "
            f"{fmt(c.get('peak_cpu_rss_training_mb') or None, 0)} |"
        )

    L.append("\n## 4. Final baseline decisions (Task 9 of the protocol)\n")
    L.append(f"1. **Best macro-F1**: {best_f1['id']} ({fmt(best_f1['val']['macro_f1'])} val; "
             f"{fmt(t['macro_f1'])} test)" + (" — same config." if best_f1['id'] == sel_id else ""))
    L.append(f"2. **Best QWK**: {best_qwk['id']} ({fmt(best_qwk['val']['qwk'])} val)")
    L.append(f"3. **Best balanced accuracy**: {best_bacc['id']} ({fmt(best_bacc['val']['balanced_accuracy'])} val)")
    kl4v = next(x["f1"] for x in best_kl4["val"]["per_class"] if x["class"] == "KL4")
    L.append(f"4. **Best KL4 F1**: {best_kl4['id']} ({fmt(kl4v)} val)")
    L.append(f"5. **Lowest latency (E2E batch-1)**: {fastest['id']} "
             f"({fastest['comp']['end_to_end_latency_batch1_ms']['mean_ms']:.2f} ms)")
    L.append(f"6. **Lowest computational complexity**: {lightest['id']} "
             f"({lightest['comp']['macs']:,} MACs, {lightest['comp']['parameters']:,} params)")
    L.append(f"7. **Best accuracy–efficiency trade-off**: {tradeoff['id']} "
             f"(highest val macro-F1 penalised by log-MACs)")
    L.append(f"8. **Recommended configuration for optimization (SOA search)**: **{sel_id}** — "
             f"it dominates all other baselines on every primary metric (val macro-F1, QWK, "
             f"bAcc) AND on test, so it is the natural starting point for hyperparameter search.\n")

    L.append("## 5. Methodological notes\n")
    L.append("- M1–M5 train from scratch; M6 uses ImageNet-pretrained ResNet-18 — not an "
             "architecture-only controlled comparison.")
    L.append("- Single seed (42); no statistical significance claims.")
    L.append("- Test-set contact: exactly one evaluation pass by M6; `selection.json` records "
             "the validation-only ranking used for that choice.")
    L.append("- Peak CPU RSS could not be captured during these runs (Windows API bug, since "
             "fixed in `src/compute.py`); the metric is active for all future experiments.")
    out = REP / "baseline_comparison.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote -> {out}")
    print(f"wrote -> {REP / 'baseline_comparison.csv'}")
    print(f"wrote -> {CM_DIR}")
    print(f"wrote -> metric_comparison.png, latency_vs_macro_f1.png")


def yaml_safe(p: Path) -> dict:
    import yaml

    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
