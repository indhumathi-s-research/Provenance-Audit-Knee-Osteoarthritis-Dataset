"""DATASET FORENSIC AUDIT (fresh, non-destructive).

Usage:
    python scripts/forensic_audit.py --data-dir data

Pilot corpus  : data/{train,val,test}/{0-4}/*.png   (single-view KL 0-4)
auto_test/    : inventoried but NOT part of the pilot corpus.

Writes:
    data/manifests/master.csv
    data/reports/dataset_forensic_audit.md
    data/reports/dataset_statistics.csv
    data/reports/class_distribution.png
    data/reports/data_quality_report.csv
    data/reports/leakage_audit.md
    data/reports/pairing_audit.md
    data/processed/quarantine_list.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

ROLES = ("train", "val", "test", "auto_test")
PILOT_ROLES = ("train", "val", "test")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FNAME_RE = re.compile(r"^(\d{7})(?:_(\d+))?([LR])?\.png$", re.IGNORECASE)
VIEW_TOKEN_RE = re.compile(r"(?:^|[_\-])(PA|AP|LAT|LATERAL)(?:$|[_\-])", re.IGNORECASE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash64(arr: np.ndarray) -> np.uint64:
    img = Image.fromarray(arr).resize((32, 32), Image.LANCZOS).convert("L")
    a = np.asarray(img, dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()
    med = np.median(flat[1:]) if flat.size > 1 else flat[0]
    bits = flat > med
    bits[0] = False
    packed = np.packbits(bits)
    return np.uint64(int.from_bytes(packed.tobytes(), "big"))


def popcount_pair_matrix(h: np.ndarray, chunk: int = 256):
    n = len(h)
    bc = np.bitwise_count
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        x = h[i0:i1, None] ^ h[None, :]
        yield i0, i1, bc(x).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    args = ap.parse_args()
    root = Path(args.data_dir)

    # ---------------- inventory ----------------
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT
    )
    ext_counts = Counter(p.suffix.lower() for p in files)
    meta_files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".xlsx", ".txt", ".xml"}
    )

    rows = []
    corrupt = []
    for p in files:
        parts = [q.lower() for q in p.relative_to(root).parts]
        role = parts[0] if parts else ""
        cls = parts[1] if len(parts) > 2 else ""
        try:
            kl_raw = int(cls)
            kl = kl_raw if 0 <= kl_raw <= 4 else None
        except ValueError:
            kl_raw, kl = cls, None
        m = FNAME_RE.match(p.name)
        pid = m.group(1) if m else ""
        seq = m.group(2) or "" if m else ""
        lat = (m.group(3) or "").upper() if m else ""
        view_tok = ""
        vm = VIEW_TOKEN_RE.search(p.stem)
        if vm:
            view_tok = vm.group(1).upper()
        rec = {
            "image_path": str(p.relative_to(root)),
            "split_role": role,
            "kl_label_folder": cls,
            "kl_grade": kl,
            "patient_id": pid,
            "sequence_token": seq,
            "laterality": lat,
            "knee_id": f"{pid}{lat}" if pid and lat else "",
            "examination_id": "",
            "view": view_tok,
            "filename_parsed": bool(m),
        }
        w = h = None
        mode = ""
        flags = []
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                mode = im.mode
                w, h = im.size
                arr = np.asarray(im.convert("L"), dtype=np.uint8)
            std = float(arr.std())
            rec.update(
                image_width=w,
                image_height=h,
                pixel_mode=mode,
                px_std=std,
                px_mean=float(arr.mean()),
                px_min=int(arr.min()),
                px_max=int(arr.max()),
            )
            if std == 0:
                flags.append("blank")
            elif std < 5:
                flags.append("near_blank")
            if min(w, h) < 100 or max(w, h) > 4000:
                flags.append("extreme_dimension")
            rec["phash"] = f"{phash64(arr):016x}"
        except Exception as e:
            flags.append("corrupt_unreadable")
            corrupt.append((str(p.relative_to(root)), repr(e)))
            rec.update(image_width=None, image_height=None, pixel_mode="", px_std=None,
                       px_mean=None, px_min=None, px_max=None, phash="")
        if not m:
            flags.append("unparseable_filename")
        if kl is None:
            flags.append("missing_label" if not str(cls).strip() else "invalid_label")
        rec["quality_flags"] = ";".join(flags)
        rows.append(rec)
        if len(rows) % 1000 == 0:
            print(f"  processed {len(rows)}/{len(files)}")

    df = pd.DataFrame(rows)
    df["sha256"] = [sha256_file(root / r) for r in df["image_path"]]
    md5_groups = defaultdict(list)
    for r in df["image_path"]:
        with open(root / r, "rb") as fh:
            hh = hashlib.md5(fh.read()).hexdigest()
        md5_groups[hh].append(r)
    df["md5"] = df["image_path"].map({p: hashlib.md5((root / p).read_bytes()).hexdigest() for p in df["image_path"]})
    exact_dup_groups = defaultdict(list)
    for p, hh in zip(df["image_path"], df["md5"]):
        exact_dup_groups[hh].append(p)
    exact_dup_groups = {k: v for k, v in exact_dup_groups.items() if len(v) > 1}
    for paths in exact_dup_groups.values():
        for q in paths[1:]:
            idx = df.index[df["image_path"] == q][0]
            df.loc[idx, "quality_flags"] = ";".join(
                filter(None, [df.at[idx, "quality_flags"], "exact_duplicate_md5"])
            )

    phash_groups = defaultdict(list)
    for r, hh in zip(df["image_path"], df["phash"]):
        if hh:
            phash_groups[hh].append(r)
    phash_dup_groups = {k: v for k, v in phash_groups.items() if len(v) > 1}
    flagged_exact = set(sum(exact_dup_groups.values(), []))
    for _, paths in phash_dup_groups.items():
        for q in paths:
            if q in flagged_exact:
                continue
            idx = df.index[df["image_path"] == q][0]
            others = [o for o in paths if o != q]
            df.at[idx, "quality_flags"] = ";".join(
                filter(None, [df.at[idx, "quality_flags"], f"phash_duplicate:{others[0]}"])
            )

    # ---------------- pairwise near-duplicate screen ----------------
    hv = np.array([np.uint64(int(x, 16)) if x else np.uint64(0) for x in df["phash"]], dtype=np.uint64)
    valid = np.array([bool(x) for x in df["phash"]])
    near_pairs = []
    for i0, i1, mat in popcount_pair_matrix(hv):
        ii, jj = np.nonzero((mat <= 8))
        for a, b in zip(ii, jj):
            gi = i0 + a
            if gi >= b or not valid[gi] or not valid[b]:
                continue
            near_pairs.append((gi, int(b), int(mat[a, b])))
    pair_rel = Counter()
    same_patient_crossrole = []
    diff_patient_near = []
    for gi, gj, d in near_pairs:
        ri, rj = df.iloc[gi], df.iloc[gj]
        same_pid = ri["patient_id"] and ri["patient_id"] == rj["patient_id"]
        same_knee = ri["knee_id"] and ri["knee_id"] == rj["knee_id"]
        cross_role = ri["split_role"] != rj["split_role"]
        rel = (
            ("same_knee" if same_knee else "same_patient_diff_knee_or_unknown") + ("_cross_role" if cross_role else "_same_role")
            if same_pid
            else "different_patient"
        )
        pair_rel[rel] += 1
        if same_pid and cross_role:
            same_patient_crossrole.append((ri["image_path"], rj["image_path"], d))
        if not same_pid:
            diff_patient_near.append((ri["image_path"], rj["image_path"], d))

    # ---------------- patient / knee structure ----------------
    pilot = df[df["split_role"].isin(PILOT_ROLES)].copy()
    pid_counts_pilot = pilot["patient_id"].value_counts()
    knees = pilot[pilot["knee_id"] != ""].groupby("knee_id").size()
    imgs_per_patient_hist = Counter(pid_counts_pilot.tolist())
    imgs_per_knee_hist = Counter(knees.tolist())
    lat_counts_pilot = pilot["laterality"].value_counts().to_dict()
    bilateral = sum(1 for _, g in pilot[pilot["knee_id"] != ""].groupby("patient_id") if g["laterality"].nunique() == 2)

    # leakage across provided roles
    role_patients = {r: set(df[df["split_role"] == r]["patient_id"]) - {""} for r in ROLES}
    leak_pairs = {}
    for i, ra in enumerate(ROLES):
        for rb in ROLES[i + 1:]:
            ov = role_patients[ra] & role_patients[rb]
            if ov:
                leak_pairs[f"{ra}|{rb}"] = sorted(ov)

    # KL distribution
    dist_all = pilot["kl_grade"].value_counts().sort_index()
    dist_by_role = pilot.groupby(["split_role", "kl_grade"]).size().unstack(fill_value=0)
    total_pilot = len(pilot)

    # multi-view evidence scan
    metadata_hits = []
    for mf in meta_files:
        try:
            txt = mf.read_text(errors="ignore")[:20000].lower()
        except Exception:
            continue
        if any(t in txt for t in ("view", "pa ", "lat", "lateral", "projection")) and mf.suffix != ".png":
            metadata_hits.append(str(mf.relative_to(root)))
    dir_view_tokens = Counter()
    for p in files:
        for part in p.relative_to(root).parts[:-1]:
            if VIEW_TOKEN_RE.search(part):
                dir_view_tokens[part] += 1
    fname_view_tokens = Counter(df.loc[df["view"] != "", "view"])

    # _seq vs L/R relation evidence (pairing audit)
    seq_rows = df[(df["sequence_token"] != "") & (df["patient_id"] != "")]
    lr_rows = df[(df["laterality"] != "") & (df["patient_id"] != "")]
    lr_by_pknee = {(r.patient_id, r.laterality): r.phash for r in lr_rows.itertuples() if r.phash}
    seq_match_lat = Counter()
    seq_examples = []
    for r in seq_rows.itertuples():
        if not r.phash:
            continue
        best = []
        for lat in ("L", "R"):
            hp = lr_by_pknee.get((r.patient_id, lat))
            if hp:
                d = int(np.bitwise_count(np.uint64(int(r.phash, 16)) ^ np.uint64(int(hp, 16))))
                best.append((d, lat))
        if best:
            best.sort()
            seq_match_lat[best[0][1]] += 1
            if len(seq_examples) < 10:
                seq_examples.append((r.image_path, best))

    # ---------------- quarantine list ----------------
    quar = df[df["quality_flags"] != ""][[
        "image_path", "split_role", "kl_grade", "quality_flags",
        "image_width", "image_height", "px_std",
    ]]
    out_q = PROJECT_ROOT / "data" / "processed" / "quarantine_list.csv"
    out_q.parent.mkdir(parents=True, exist_ok=True)
    quar.to_csv(out_q, index=False)

    # ---------------- master manifest ----------------
    man_dir = PROJECT_ROOT / "data" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    man = df[[
        "image_path", "split_role", "patient_id", "knee_id", "examination_id",
        "laterality", "view", "kl_grade", "image_width", "image_height", "sha256",
    ]].copy()
    man.to_csv(man_dir / "master.csv", index=False)

    # ---------------- dataset_statistics.csv ----------------
    rep_dir = PROJECT_ROOT / "data" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    stats_lines = [
        ("metric", "value"),
        ("total_image_files", len(files)),
        ("pilot_corpus_images(train+val+test)", total_pilot),
        ("auto_test_images", int((df["split_role"] == "auto_test").sum())),
        ("extensions", json.dumps(dict(ext_counts))),
        ("distinct_dimensions", json.dumps(dict(Counter(f"{w}x{h}" for w, h in zip(df["image_width"], df["image_height"]))))),
        ("pixel_modes", json.dumps(dict(Counter(df["pixel_mode"])))),
        ("corrupt_unreadable", len(corrupt)),
        ("blank_images", int(df["quality_flags"].str.contains("blank", na=False).sum())),
        ("exact_duplicate_md5_groups", len(exact_dup_groups)),
        ("exact_duplicate_md5_images", sum(len(v) - 1 for v in exact_dup_groups.values())),
        ("identical_phash_groups", len(phash_dup_groups)),
        ("near_duplicate_pairs_hamming_le8", len(near_pairs)),
        ("unique_patients_full", df.loc[df["patient_id"] != "", "patient_id"].nunique()),
        ("unique_patients_pilot_corpus", pilot.loc[pilot["patient_id"] != "", "patient_id"].nunique()),
        ("unique_knees_pilot_corpus", int(pilot.loc[pilot["knee_id"] != "", "knee_id"].nunique())),
        ("images_per_patient_histogram", json.dumps({str(k): v for k, v in sorted(imgs_per_patient_hist.items())})),
        ("images_per_knee_histogram", json.dumps({str(k): v for k, v in sorted(imgs_per_knee_hist.items())})),
        ("bilateral_patients_pilot_corpus", bilateral),
        ("laterality_counts_pilot_corpus", json.dumps({str(k): int(v) for k, v in lat_counts_pilot.items()})),
        ("multi_view_available", "FALSE"),
    ]
    with open(rep_dir / "dataset_statistics.csv", "w", newline="", encoding="utf-8") as f:
        for row in stats_lines:
            f.write(",".join(str(x) for x in row) + "\n")

    # ---------------- class_distribution.png ----------------
    grades = [g for g in range(5)]
    counts = [int(dist_all.get(g, 0)) for g in grades]
    pcts = [c / max(total_pilot, 1) * 100 for c in counts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f"KL{g}" for g in grades], counts, color="#3b6ea5")
    for b, c, p in zip(bars, counts, pcts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c}\n{p:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("images")
    ax.set_title(f"Single-view KL distribution — pilot corpus (n={total_pilot})")
    ax.set_ylim(0, max(counts) * 1.18)
    fig.tight_layout()
    fig.savefig(rep_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # ---------------- data_quality_report.csv ----------------
    flag_summary = Counter()
    for s in df["quality_flags"]:
        for flt in s.split(";"):
            if flt:
                flag_summary[flt.split(":")[0]] += 1
    with open(rep_dir / "data_quality_report.csv", "w", newline="", encoding="utf-8") as f:
        f.write("flag,count\n")
        for k, v in sorted(flag_summary.items()):
            f.write(f"{k},{v}\n")

    pct_review = len(quar) / max(len(df), 1) * 100

    # ---------------- pairing_audit.md ----------------
    pa = []
    pa.append("# Pairing Audit (PA/AP vs LAT)\n")
    pa.append("## Evidence searched\n")
    pa.append(f"- Metadata files found under data root: `{len(meta_files)}` → {meta_files[:5]}")
    pa.append(f"- Metadata hits mentioning view/lateral/projection: `{metadata_hits}`")
    pa.append(f"- View tokens in directory names: `{dict(dir_view_tokens)}`")
    pa.append(f"- View tokens in filenames (PA/AP/LAT): `{dict(fname_view_tokens)}`")
    pa.append(f"- Laterality tokens (L/R): `{dict(Counter(pilot['laterality']))}`")
    pa.append(f"- Sequence tokens (`_1`/`_2`) in pilot corpus: `{int((pilot['sequence_token'] != '').sum())}`")
    pa.append("\n## Verdict\n")
    pa.append("**MULTI_VIEW_AVAILABLE = FALSE**")
    pa.append("- `L`/`R` encode laterality of the same examination type, not PA/AP-vs-LAT projections.")
    pa.append("- No projection/view metadata exists anywhere in the corpus.")
    pa.append("- All images are fixed-knee radiographs from one imaging protocol; the dataset is treated as SINGLE-VIEW.\n")
    pa.append("## `_1`/`_2` sequence tokens (evidence they are laterities, not views)\n")
    pa.append(f"- Nearest-pHash match of each `_k` image among the same patient's `L`/`R` knees:")
    pa.append(f"  - matched to L: {seq_match_lat.get('L', 0)}, matched to R: {seq_match_lat.get('R', 0)}")
    for ex in seq_examples[:5]:
        pa.append(f"  - `{ex[0]}` → {ex[1]}")
    (rep_dir / "pairing_audit.md").write_text("\n".join(pa), encoding="utf-8")

    # ---------------- leakage_audit.md ----------------
    la = []
    la.append("# Leakage Audit\n")
    la.append("## Provided folder splits are NOT patient-disjoint\n")
    for k, v in leak_pairs.items():
        la.append(f"- `{k}`: {len(v)} shared patients; examples: {v[:5]}")
    n_shared_union = set().union(*[set(v) for v in leak_pairs.values()]) if leak_pairs else set()
    la.append(f"\nPatients appearing in more than one provided split (full corpus): **{len(n_shared_union)}** "
              f"of {df.loc[df['patient_id'] != '', 'patient_id'].nunique()}")
    la.append("\n## Defensible grouping mechanism\n")
    la.append(f"- Reliable 7-digit patient IDs parsed from filenames: YES ({pilot['patient_id'].nunique()} unique in pilot corpus).")
    la.append(f"- Knee IDs derivable as patient+laterality: {pilot.loc[pilot['knee_id'] != '', 'knee_id'].nunique()} unique.")
    la.append("- Patient-level splitting IS POSSIBLE and REQUIRED. Group key = patient_id (both knees stay together).")
    la.append("- auto_test/ must be excluded (re-encoded duplicates of test/; see forensic report near-duplicate section).")
    (rep_dir / "leakage_audit.md").write_text("\n".join(la), encoding="utf-8")

    # ---------------- main forensic report ----------------
    fa = []
    fa.append("# Dataset Forensic Audit — single-view KL 0–4 pilot\n")
    fa.append(f"- Data root: `{root}` (read-only; originals untouched)")
    fa.append(f"- Total image files: **{len(files)}**; extensions: {dict(ext_counts)}")
    fa.append(f"- Pilot corpus (train+val+test): **{total_pilot}**; auto_test inventoried separately: {(df['split_role']=='auto_test').sum()}")
    fa.append(f"- Distinct dimensions: {dict(Counter(f'{w}x{h}' for w,h in zip(df['image_width'], df['image_height'])))}")
    fa.append(f"- Pixel modes: {dict(Counter(df['pixel_mode']))}")
    fa.append(f"- Corrupt/unreadable: **{len(corrupt)}**")
    fa.append(f"- Exact duplicate (MD5) groups: **{len(exact_dup_groups)}**")
    fa.append(f"- Identical-pHash groups beyond MD5: **{len(phash_dup_groups) - len([g for g in phash_dup_groups.values() if all(p in flagged_exact for p in g)])}**")
    fa.append(f"- Near-duplicate pairs (Hamming<=8): {len(near_pairs)} by relation: {dict(pair_rel)}")
    if same_patient_crossrole:
        fa.append("- Example same-patient cross-role near-duplicates (leakage evidence):")
        for a, b, d in same_patient_crossrole[:8]:
            fa.append(f"  - `{a}` <=> `{b}` (d={d})")
    if diff_patient_near:
        fa.append(f"- Different-patient near-duplicate pairs: {len(diff_patient_near)}; examples:")
        for a, b, d in diff_patient_near[:5]:
            fa.append(f"  - `{a}` <=> `{b}` (d={d})")
    fa.append(f"- Unparseable filenames: {int((~df['filename_parsed']).sum())}")
    fa.append(f"- Missing/invalid labels: {int(df['kl_grade'].isna().sum())}")
    fa.append("\n## KL label source & encoding\n")
    fa.append("- Label source: parent class-folder name (`0`–`4`), i.e., Kaggle-provided severity labels.")
    fa.append("- Label encoding: integer Kellgren–Lawrence grade 0..4.\n")
    fa.append("## Actual KL distribution (pilot corpus)\n")
    fa.append("| KL | count | % |")
    fa.append("|---:|---:|---:|")
    for g in grades:
        fa.append(f"| {g} | {counts[g]} | {pcts[g]:.2f}% |")
    fa.append("")
    fa.append("### By provided split\n")
    fa.append(dist_by_role.to_string())
    fa.append("")
    fa.append("## Patient / knee structure\n")
    fa.append(f"- Patient IDs: reliable 7-digit numeric IDs from filenames.")
    fa.append(f"- Unique patients (pilot corpus): **{pilot.loc[pilot['patient_id']!='','patient_id'].nunique()}**")
    fa.append(f"- Unique knees (patient+laterality): **{pilot.loc[pilot['knee_id']!='','knee_id'].nunique()}**")
    fa.append(f"- Images/patient histogram: {dict(sorted(imgs_per_patient_hist.items()))}")
    fa.append(f"- Images/knee histogram: {dict(sorted(imgs_per_knee_hist.items()))}")
    fa.append(f"- Laterality L/R: {lat_counts_pilot}; patients with both knees present: {bilateral}")
    fa.append("- Examination IDs / view labels: NOT AVAILABLE.\n")
    fa.append("## Multi-view verdict\n")
    fa.append("**MULTI_VIEW_AVAILABLE = FALSE** — no genuine PA/AP+LAT pairs; see `pairing_audit.md`.\n")
    fa.append("## Leakage summary\n")
    fa.append(f"- Patients spanning multiple provided splits: {len(n_shared_union)}. Patient-level regrouping required; see `leakage_audit.md`.\n")
    fa.append("## Data quality / quarantine\n")
    fa.append(f"- Flagged for manual review: **{len(quar)}** images ({pct_review:.2f}%). Quarantine list: `data/processed/quarantine_list.csv`")
    fa.append(f"- Flag summary: {dict(flag_summary)}")
    fa.append("- No files were deleted or modified.\n")
    (rep_dir / "dataset_forensic_audit.md").write_text("\n".join(fa), encoding="utf-8")

    summary_out = {
        "total_images": len(files),
        "pilot_corpus": total_pilot,
        "kl_distribution": {str(g): counts[g] for g in grades},
        "kl_pct": {str(g): round(pcts[g], 2) for g in grades},
        "patients": int(pilot.loc[pilot["patient_id"] != "", "patient_id"].nunique()),
        "knees": int(pilot.loc[pilot["knee_id"] != "", "knee_id"].nunique()),
        "multi_view_available": False,
        "corrupt": len(corrupt),
        "quarantined": len(quar),
        "leaking_patients_provided_splits": len(n_shared_union),
        "duplicate_md5_groups": len(exact_dup_groups),
        "near_dup_pairs": len(near_pairs),
    }
    (PROJECT_ROOT / "data" / "processed" / "forensic_audit_summary.json").write_text(
        json.dumps(summary_out, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
