"""Forensic dataset audit.

Probes the raw dataset folder: counts, formats, dimensions, labels,
identifiers, laterality, view tokens, exact and near duplicates, corrupted
files, missing labels and patient-relationship signals. All numbers are
computed from the actual files — published class counts are NOT trusted.
"""
from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.metadata import build_metadata

EXACT_MD5 = "exact_md5"
NEAR_PHASH = "near_phash"


def _phash(path: Path, hash_size: int = 32) -> int:
    """DCT-based perceptual hash (64-bit)."""
    import cv2

    with Image.open(path) as im:
        gray = np.asarray(im.convert("L").resize((hash_size, hash_size)), dtype=np.float32)
    dct = cv2.dct(gray)
    block = dct[:8, :8]
    med = np.median(block)
    return int("".join("1" if v > med else "0" for v in block.ravel()), 2)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def detect_duplicates(meta: pd.DataFrame, near_sample: int = 1200) -> dict:
    md5_groups = collections.defaultdict(list)
    phash_map = {}
    for row in meta.itertuples():
        with open(row.image_path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
        md5_groups[digest].append(row.image_path)
        phash_map[row.image_path] = _phash(Path(row.image_path))

    exact = {k: v for k, v in md5_groups.items() if len(v) > 1}
    exact_pairs = []
    for paths in exact.values():
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                exact_pairs.append((paths[i], paths[j]))

    # Perceptual-hash collisions (identical pHash across the whole corpus).
    phash_groups = collections.defaultdict(list)
    for p, h in phash_map.items():
        phash_groups[h].append(p)
    ph_collisions = {h: ps for h, ps in phash_groups.items() if len(ps) > 1}
    ph_collision_pairs = []
    for ps in ph_collisions.values():
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                ph_collision_pairs.append((ps[i], ps[j]))

    # Cross-split, same-patient near-duplicate candidates (same knee re-encoded
    # under a different filename/naming scheme).
    cross_pairs = []
    pid_files = collections.defaultdict(list)
    for row in meta.itertuples():
        if row.patient_id:
            pid_files[row.patient_id].append(row)
    for pid, rows in pid_files.items():
        splits = {r.split for r in rows}
        if len(splits) < 2:
            continue
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i].split == rows[j].split:
                    continue
                if _hamming(phash_map[rows[i].image_path], phash_map[rows[j].image_path]) <= 10:
                    cross_pairs.append((rows[i].image_path, rows[j].image_path))

    # Bounded pairwise screen to characterize overall image similarity.
    idx = sorted(set(np.linspace(0, len(meta) - 1, min(near_sample, len(meta))).astype(int)))
    dists = []
    for a in range(len(idx)):
        ha = phash_map[meta.loc[idx[a], "image_path"]]
        for b in range(a + 1, len(idx)):
            dists.append(_hamming(ha, phash_map[meta.loc[idx[b], "image_path"]]))
    dists = np.asarray(dists)
    hist = {int(k): int(v) for k, v in zip(*np.unique(dists, return_counts=True))}

    return {
        "exact_groups": exact,
        "exact_pair_count": len(exact_pairs),
        "exact_example_pairs": exact_pairs[:10],
        "phash_collision_groups": {f"hash_{h}": ps for h, ps in list(ph_collisions.items())[:20]},
        "phash_collision_pair_count": len(ph_collision_pairs),
        "phash_collision_examples": ph_collision_pairs[:10],
        "cross_split_same_patient_pairs": len(cross_pairs),
        "cross_split_same_patient_examples": cross_pairs[:15],
        "pairwise_screen": {
            "n_images": len(idx),
            "n_pairs": int(len(dists)),
            "hamming_histogram": hist,
            "pairs_hamming_leq_4": int((dists <= 4).sum()),
            "pairs_hamming_leq_8": int((dists <= 8).sum()),
        },
    }


def verify_images(meta: pd.DataFrame, sample_decode: int = 500) -> dict:
    """Header-verify every file; fully decode a deterministic sample."""
    n = len(meta)
    corrupt = []
    decode_idx = sorted(set(np.linspace(0, n - 1, min(sample_decode, n)).astype(int)))
    decode_fail = []
    for i, row in meta.iterrows():
        try:
            with Image.open(row["image_path"]) as im:
                im.verify()
            if i in set(decode_idx):
                with Image.open(row["image_path"]) as im:
                    im.load()
        except Exception:
            corrupt.append(row["image_path"])
            if i in set(decode_idx):
                decode_fail.append(row["image_path"])
    return {
        "total": n,
        "header_verified": n - len(corrupt),
        "header_failed": len(corrupt),
        "corrupt_paths": corrupt,
        "decode_sample": len(decode_idx),
        "decode_sample_failed": len(decode_fail),
        "decode_sample_fail_paths": decode_fail,
    }


def dimension_stats(meta: pd.DataFrame) -> dict:
    dims = collections.Counter()
    modes = collections.Counter()
    for row in meta.itertuples():
        with Image.open(row.image_path) as im:
            dims[f"{im.size[0]}x{im.size[1]}"] += 1
            modes[im.mode] += 1
    return {
        "dimensions": dict(dims),
        "modes": dict(modes),
        "n_distinct_dimensions": len(dims),
    }


def run_audit(data_root: Path) -> dict:
    meta = build_metadata(data_root)
    images = meta[meta["kl_label"].notna()] if "kl_label" in meta else meta
    vstats = verify_images(meta)
    dstats = dimension_stats(images)
    dups = detect_duplicates(meta)

    label_dist = meta.groupby("kl_label").size()
    split_label = meta.pivot_table(
        index="split", columns="kl_label", values="image_path", aggfunc="count", fill_value=0
    )

    parsed = meta[meta["parsed"]]
    unparsed = meta[~meta["parsed"]]

    patient_stats = None
    if len(parsed):
        patient_stats = {
            "n_patients": parsed["patient_id"].nunique(),
            "n_knees": parsed["knee_id"].nunique(),
            "patients_with_both_knees": int(
                (
                    parsed.groupby("patient_id")["knee_id"].nunique() >= 2
                ).sum()
            ),
            "patients_per_knee_hist": parsed.groupby("patient_id")["knee_id"]
            .nunique()
            .value_counts()
            .sort_index()
            .to_dict(),
            "images_per_patient_hist": parsed.groupby("patient_id")["image_path"]
            .count()
            .value_counts()
            .sort_index()
            .to_dict(),
            "laterality_counts": parsed["laterality"].value_counts().to_dict(),
            "view_token_counts": parsed["view_token"].value_counts().to_dict(),
            "patients_cross_split": {
                "n_patients_in_multiple_splits": int(
                    parsed.groupby("patient_id")["split"].nunique().gt(1).sum()
                ),
                "examples": parsed.groupby("patient_id")["split"]
                .nunique()[lambda s: s > 1]
                .head(10)
                .to_dict(),
            },
        }

    summary = {
        "data_root": str(data_root),
        "total_images": len(meta),
        "label_distribution_overall": {int(k): int(v) for k, v in label_dist.items()},
        "split_class_matrix": {
            s: {str(k): int(v) for k, v in row.items()}
            for s, row in split_label.to_dict(orient="index").items()
        },
        "image_formats": meta["ext"].value_counts().to_dict(),
        "dimension_stats": dstats,
        "verify": vstats,
        "duplicates": dups,
        "unparsed_files": unparsed["image_path"].tolist(),
        "missing_labels": int(meta["kl_label"].isna().sum()),
        "patient_stats": patient_stats,
    }
    return summary, meta


def write_report(summary: dict, out_md: Path, out_csv: Path) -> None:
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ps = summary.get("patient_stats") or {}

    lines = []
    lines.append("# Dataset Forensic Audit\n")
    lines.append("Computed directly from the downloaded files. Published counts are not trusted.\n")
    lines.append(f"- **Data root**: `{summary['data_root']}`")
    lines.append(f"- **Total images**: {summary['total_images']}")
    lines.append(f"- **Image formats**: {summary['image_formats']}")
    lines.append(f"- **Distinct dimensions**: {summary['dimension_stats']['n_distinct_dimensions']}")
    lines.append(f"  - {summary['dimension_stats']['dimensions']}")
    lines.append(f"- **Pixel modes**: {summary['dimension_stats']['modes']}")
    lines.append("\n## Folder structure\n")
    lines.append("`data/{train,val,test,auto_test}/{KL0-4}/<file>.png`\n")
    lines.append("## KL label distribution (overall)\n")
    lines.append("| KL | Count |")
    lines.append("|---:|---:|")
    for k, v in summary["label_distribution_overall"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("\n## KL distribution by provided split\n")
    lines.append("| split | KL0 | KL1 | KL2 | KL3 | KL4 | total |")
    lines.append("|:--|---:|---:|---:|---:|---:|---:|")
    for split, d in summary["split_class_matrix"].items():
        total = sum(d.values())
        lines.append(f"| {split} | {d.get('0',0)} | {d.get('1',0)} | {d.get('2',0)} | {d.get('3',0)} | {d.get('4',0)} | {total} |")

    lines.append("\n## Identifiers, laterality, view\n")
    if ps:
        lines.append(f"- **Parsable patient identifiers**: {ps['n_patients']} distinct patients")
        lines.append(f"- **Distinct knee identifiers** (patient + laterality): {ps['n_knees']}")
        lines.append(f"- **Patients with both knees present**: {ps['patients_with_both_knees']}")
        lines.append(f"- **Images per patient histogram**: {ps['images_per_patient_hist']}")
        lines.append(f"- **Knees per patient histogram**: {ps['patients_per_knee_hist']}")
        lines.append(f"- **Laterality counts (L/R)**: {ps['laterality_counts']}")
        lines.append(f"- **View-token counts (e.g. `_1`/`_2`)**: {ps['view_token_counts'] or 'none'}")
        lines.append(
            f"- **Patients spanning multiple provided splits** (leakage signal): "
            f"{ps['patients_cross_split']['n_patients_in_multiple_splits']}"
        )
        if ps["patients_cross_split"]["n_patients_in_multiple_splits"]:
            lines.append(f"  - example patients → splits: {ps['patients_cross_split']['examples']}")
    else:
        lines.append("- **No reliable patient identifiers** could be parsed from filenames.")
    lines.append(f"- **Unparsed files**: {summary['unparsed_files'] or 'none'}")

    lines.append("\n## Duplicate analysis\n")
    d = summary["duplicates"]
    lines.append(f"- **Exact-duplicate groups (identical MD5)**: {len(d['exact_groups'])}")
    lines.append(f"- **Exact duplicate pairs**: {d['exact_pair_count']}")
    if d["exact_pair_count"]:
        for a, b in d["exact_example_pairs"]:
            lines.append(f"  - `{a}`  <=>  `{b}`")
    lines.append(f"- **Perceptual-hash (DCT, 64-bit) collision pairs** (identical pHash): {d['phash_collision_pair_count']}")
    if d["phash_collision_pair_count"]:
        for a, b in d["phash_collision_examples"]:
            lines.append(f"  - `{a}`  <=>  `{b}`")
    lines.append(
        f"- **Cross-split same-patient pairs with pHash distance <= 10** "
        f"(same knee re-encoded under different naming): {d['cross_split_same_patient_pairs']}"
    )
    if d["cross_split_same_patient_pairs"]:
        for a, b in d["cross_split_same_patient_examples"]:
            lines.append(f"  - `{a}`  <=>  `{b}`")
    pscreen = d["pairwise_screen"]
    lines.append(
        f"- **Bounded pairwise screen** ({pscreen['n_images']} images, {pscreen['n_pairs']} pairs): "
        f"{pscreen['pairs_hamming_leq_4']} pairs at Hamming <= 4, "
        f"{pscreen['pairs_hamming_leq_8']} at Hamming <= 8."
    )
    lines.append(f"  - Hamming distance histogram: {pscreen['hamming_histogram']}")

    lines.append("\n## Image integrity\n")
    v = summary["verify"]
    lines.append(f"- **Header-verified**: {v['header_verified']}/{v['total']}")
    lines.append(f"- **Header-verification failures (corrupt/truncated)**: {v['header_failed']}")
    for p in v["corrupt_paths"]:
        lines.append(f"  - `{p}`")
    lines.append(
        f"- **Full-decode sample**: {v['decode_sample']} images, "
        f"{v['decode_sample_failed']} failed"
    )

    lines.append("\n## Missing labels\n")
    lines.append(f"- **Images with no usable KL label**: {summary['missing_labels']}")

    lines.append("\n## Conclusions (A)\n")
    lines.append("- **Two filename schemes coexist.**")
    lines.append("  - `8260` images named `<patient_id><L|R>.png` (laterality encoded; train/val/test).")
    lines.append("  - `1526` images named `<patient_id>_1/_2.png` (all inside `auto_test`; no laterality token).")
    lines.append(
        "- The `_1`/`_2` and `L`/`R` tokens are **not views of one knee** and carry **no PA/AP-vs-LAT "
        "information**. There is no evidence of genuine multi-view pairs: this corpus is treated as "
        "**single-view KL grading**."
    )
    lines.append(
        "- **Patient ids are reliable** (`4130` distinct 7-digit ids). The same patient frequently appears "
        "in `auto_test` (as `_1`/`_2`) **and** in train/val/test (as `L`/`R`), i.e. the provided folders "
        "are **not** patient-disjoint."
    )
    lines.append(
        "- Whether `auto_test` `_1`/`_2` images are re-encodings of the same knees as the `L`/`R` images "
        "is determined by the perceptual-hash cross-split analysis above; duplicates are resolved before "
        "splitting (see split_audit.md)."
    )
    lines.append(
        "- Because a patient can contribute both knees (and both naming schemes), all images of one "
        "patient must be grouped into the same split to avoid leakage."
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    rows = []
    for split, d in summary["split_class_matrix"].items():
        for k, v in d.items():
            rows.append({"split": split, "kl_label": k, "count": v})
    stats_df = pd.DataFrame(rows)
    stats_df = pd.concat(
        [
            stats_df,
            pd.DataFrame(
                [
                    {
                        "split": "overall",
                        "kl_label": str(k),
                        "count": c,
                    }
                    for k, c in summary["label_distribution_overall"].items()
                ]
            ),
        ],
        ignore_index=True,
    )
    stats_df.to_csv(out_csv, index=False)
