"""Reproducible dataset-integrity audit for the distributed knee-OA KL-grading dataset.

Regenerates every dataset fact intended for the manuscript from the raw files and the
manifest, and writes:
    research/results/dataset_integrity_audit.json      (machine-readable, all numbers)
    research/results/dataset_integrity_audit.md        (human-readable companion)
    research/results/auto_test_test_label_disagreements.csv

Facts covered: image and patient counts; class distributions; train/val/test patient
overlap; auto_test <-> test pairing; exact (SHA-256) and near-duplicate (64-bit DCT
pHash) statistics; label agreement of paired auto_test/test knees; bilateral (L/R)
label agreement; patient-disjointness of splits_v1.

Nothing here trains a model or touches model outputs. Read-only on data/.

Usage:
    python scripts/dataset_integrity_audit.py
    python scripts/dataset_integrity_audit.py --data-dir data --out-dir research/results
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_FOLDERS = ["train", "val", "test"]
PHASH_THRESHOLDS = (0, 4, 8, 10)
RE_CORPUS = re.compile(r"^(\d{7})([LR])\.png$")
RE_AUTO = re.compile(r"^(\d{7})_(\d)\.png$")
AUTO_MAP = {"1": "R", "2": "L"}          # inferred; evidence reported below
AUTO_MAP_SWAPPED = {"1": "L", "2": "R"}   # control


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phash64(arr: np.ndarray) -> np.uint64:
    """Same definition as scripts/forensic_audit.py::phash64 (64-bit DCT pHash)."""
    img = Image.fromarray(arr).resize((32, 32), Image.LANCZOS).convert("L")
    a = np.asarray(img, dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()
    med = np.median(flat[1:]) if flat.size > 1 else flat[0]
    bits = flat > med
    bits[0] = False
    return np.uint64(int.from_bytes(np.packbits(bits).tobytes(), "big"))


def git_state() -> dict:
    def run(*a):
        try:
            r = subprocess.run(["git", "-C", str(PROJECT_ROOT), *a], capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    return {"head": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def class_dist(s: pd.Series) -> dict:
    vc = s.astype(int).value_counts().reindex(range(5), fill_value=0)
    return {str(k): int(v) for k, v in vc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "research" / "results"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifests" / "master.csv"
    splits_path = data_dir / "processed" / "splits_v1.csv"
    freeze_path = data_dir / "processed" / "KOA_KL_PILOT_v1_freeze.json"

    m = pd.read_csv(manifest_path, dtype={"patient_id": str})
    m["path"] = m["image_path"].str.replace("\\", "/", regex=False)
    m["folder"] = m["split_role"]
    m["kl"] = m["kl_grade"].astype(int)

    # ---- parse patient / laterality from file names ------------------------------
    pid, lat, sfx = [], [], []
    for p, f in zip(m["path"], m["folder"]):
        name = p.split("/")[-1]
        if f == "auto_test":
            g = RE_AUTO.match(name)
            pid.append(g.group(1) if g else None)
            sfx.append(g.group(2) if g else None)
            lat.append(AUTO_MAP.get(g.group(2)) if g else None)
        else:
            g = RE_CORPUS.match(name)
            pid.append(g.group(1) if g else None)
            sfx.append(None)
            lat.append(g.group(2) if g else None)
    m["pid"] = pid
    m["lat"] = lat
    m["sfx"] = sfx
    unparsed = int(m["pid"].isna().sum())
    assert (m["pid"] == m["patient_id"]).all(), "manifest patient_id disagrees with file-name patient id"

    corpus = m[m["folder"].isin(CORPUS_FOLDERS)].copy()
    auto = m[m["folder"] == "auto_test"].copy()

    res: dict = {}
    res["meta"] = {
        "script": "scripts/dataset_integrity_audit.py",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "platform": platform.platform(),
        "git": git_state(),
        "inputs": {
            "manifest": {"path": str(manifest_path.relative_to(PROJECT_ROOT)) if manifest_path.is_relative_to(PROJECT_ROOT) else str(manifest_path),
                         "sha256": sha256_file(manifest_path)},
            "splits_v1": {"path": "data/processed/splits_v1.csv", "sha256": sha256_file(splits_path)},
            "freeze": {"path": "data/processed/KOA_KL_PILOT_v1_freeze.json", "sha256": sha256_file(freeze_path)},
        },
        "definitions": {
            "patient": "7-digit patient id parsed from file name",
            "corpus": "images in provided folders train, val, test",
            "auto_test_pairing": "auto_test file <pid>_<k>.png paired to test knee <pid><lat>; k=1->R, k=2->L (inferred, see evidence)",
            "phash": "64-bit DCT pHash as in scripts/forensic_audit.py::phash64",
        },
    }

    # ---- 1. counts -----------------------------------------------------------------
    res["counts"] = {
        "total_images_all_folders": int(len(m)),
        "images_per_provided_folder": {k: int(v) for k, v in m["folder"].value_counts().reindex(["train", "val", "test", "auto_test"]).items()},
        "corpus_images_train_val_test": int(len(corpus)),
        "auto_test_images": int(len(auto)),
        "total_minus_corpus": int(len(m) - len(corpus)),
        "patients_all_folders": int(m["pid"].nunique()),
        "patients_corpus": int(corpus["pid"].nunique()),
        "patients_auto_test": int(auto["pid"].nunique()),
        "patients_per_provided_folder": {k: int(corpus[corpus.folder == k]["pid"].nunique()) for k in CORPUS_FOLDERS},
        "unparsed_file_names": unparsed,
        "images_per_patient_corpus": {str(k): int(v) for k, v in corpus.groupby("pid").size().value_counts().sort_index().items()},
        "knees_unique_patient_lat_corpus": int(corpus.groupby(["pid", "lat"]).ngroups),
        "image_dimensions": {f"{w}x{h}": int(c) for (w, h), c in Counter(zip(m["image_width"], m["image_height"])).items()},
    }

    # ---- 2. class distributions -----------------------------------------------------
    sp = pd.read_csv(splits_path, dtype={"patient_id": str})
    sp["path"] = sp["image_path"].str.replace("\\", "/", regex=False).str.replace(r"^data/", "", regex=True)
    assert set(sp["path"]) <= set(m["path"]), "splits_v1 paths not found in manifest"
    res["class_distributions"] = {
        "by_provided_folder": {k: class_dist(m[m.folder == k]["kl"]) for k in ["train", "val", "test", "auto_test"]},
        "corpus_train_val_test": class_dist(corpus["kl"]),
        "all_folders_incl_auto_test": class_dist(m["kl"]),
        "splits_v1_by_role": {r: class_dist(sp[sp.role == r]["kl_label"]) for r in ["train", "val", "test"]},
    }

    # ---- 3. patient overlap --------------------------------------------------------
    folders_by_patient_all = m.groupby("pid")["folder"].apply(lambda s: tuple(sorted(set(s))))
    folders_by_patient_core = corpus.groupby("pid")["folder"].apply(lambda s: tuple(sorted(set(s))))
    multi_all = folders_by_patient_all[folders_by_patient_all.apply(len) > 1]
    multi_core = folders_by_patient_core[folders_by_patient_core.apply(len) > 1]
    roles_by_patient = sp.groupby("patient_id")["role"].nunique()
    res["patient_overlap"] = {
        "provided_folders_train_val_test": {
            "patients": int(corpus["pid"].nunique()),
            "patients_in_more_than_one_of_train_val_test": int(len(multi_core)),
            "fraction": round(len(multi_core) / corpus["pid"].nunique(), 6),
        },
        "provided_folders_including_auto_test": {
            "patients_in_more_than_one_folder": int(len(multi_all)),
            "fraction_of_all_patients": round(len(multi_all) / m["pid"].nunique(), 6),
            "folder_combinations": {"+".join(k): int(v) for k, v in Counter(multi_all.tolist()).items()},
        },
        "auto_test_patients_not_in_test": int(len(set(auto["pid"]) - set(corpus[corpus.folder == "test"]["pid"]))),
        "test_patients_not_in_auto_test": int(len(set(corpus[corpus.folder == "test"]["pid"]) - set(auto["pid"]))),
        "splits_v1": {
            "patients": int(sp["patient_id"].nunique()),
            "patients_in_multiple_roles": int((roles_by_patient > 1).sum()),
            "images_per_role": {r: int((sp.role == r).sum()) for r in ["train", "val", "test"]},
            "patients_per_role": {r: int(sp[sp.role == r]["patient_id"].nunique()) for r in ["train", "val", "test"]},
        },
    }

    # ---- 4. exact duplicates (recomputed SHA-256) ---------------------------------
    sha_new = []
    for p in m["path"]:
        sha_new.append(sha256_file(data_dir / p))
    m["sha_new"] = sha_new
    dup_groups = m.groupby("sha_new")["path"].apply(list)
    dup_groups = dup_groups[dup_groups.apply(len) > 1]
    res["exact_duplicates"] = {
        "sha256_recomputed_from_files": True,
        "manifest_sha256_matches_recomputed": bool((m["sha_new"] == m["sha256"]).all()),
        "duplicate_groups": int(len(dup_groups)),
        "images_in_duplicate_groups": int(dup_groups.apply(len).sum()) if len(dup_groups) else 0,
    }

    # ---- 5. perceptual-hash near-duplicates -----------------------------------------
    hashes = []
    for p in m["path"]:
        arr = np.asarray(Image.open(data_dir / p).convert("L"))
        hashes.append(phash64(arr))
    H = np.array(hashes, dtype=np.uint64)
    m["phash"] = [f"{int(x):016x}" for x in H]
    n = len(m)
    folder = m["folder"].to_numpy()
    pidv = m["pid"].to_numpy()
    path_to_role = dict(zip(sp["path"], sp["role"]))
    role = np.array([path_to_role.get(p, "") for p in m["path"]])

    pairs = []  # (i, j, d) for d <= 10, i < j
    hist = np.zeros(65, dtype=np.int64)
    chunk = 256
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d = np.bitwise_count(H[i0:i1, None] ^ H[None, :]).astype(np.uint8)
        for k in range(i1 - i0):
            i = i0 + k
            row = d[k, i + 1:]
            hist += np.bincount(row, minlength=65)
            js = np.nonzero(row <= max(PHASH_THRESHOLDS))[0]
            for j in js:
                pairs.append((i, i + 1 + int(j), int(row[j])))
    P = pd.DataFrame(pairs, columns=["i", "j", "d"])
    P["same_patient"] = pidv[P.i] == pidv[P.j]
    P["fold_pair"] = ["+".join(sorted((a, b))) for a, b in zip(folder[P.i], folder[P.j])]
    P["role_pair"] = ["+".join(sorted((a, b))) if a and b else "" for a, b in zip(role[P.i], role[P.j])]

    def tally(df: pd.DataFrame, key: str) -> dict:
        out = {}
        for thr in PHASH_THRESHOLDS:
            sub = df[df.d <= thr]
            out[f"<={thr}"] = {
                "total": int(len(sub)),
                "same_patient": int(sub.same_patient.sum()),
                "different_patient": int((~sub.same_patient).sum()),
                "by_pair_type": {k: {"same_patient": int(g.same_patient.sum()), "different_patient": int((~g.same_patient).sum())}
                                 for k, g in sub.groupby(key)},
            }
        return out

    core_mask = np.isin(folder, CORPUS_FOLDERS)
    Pc = P[core_mask[P.i] & core_mask[P.j]]
    res["near_duplicates_phash"] = {
        "all_folders_by_provided_folder_pair": tally(P, "fold_pair"),
        "corpus_only_by_provided_folder_pair": tally(Pc, "fold_pair"),
        "corpus_only_by_splits_v1_role_pair": tally(Pc, "role_pair"),
        "corpus_identical_phash_pairs_d0": int((Pc.d == 0).sum()),
        "note": "Distances <= 8 between DIFFERENT patients are dominated by structurally similar knees "
                "(low-information crops); they are reported, not treated as duplicates. Same-patient pairs "
                "across provided folders are the relevant duplicate evidence.",
    }

    # ---- 6. auto_test <-> test pairing ----------------------------------------------
    test = corpus[corpus.folder == "test"]
    test_idx = {(r.pid, r.lat): idx for idx, r in test.iterrows()}
    rows, dis_rows = [], []
    for idx, r in auto.iterrows():
        tk = test_idx.get((r.pid, AUTO_MAP[r.sfx]))
        tk_sw = test_idx.get((r.pid, AUTO_MAP_SWAPPED[r.sfx]))
        rows.append((idx, tk, tk_sw))
    A = pd.DataFrame(rows, columns=["a", "t", "t_sw"])
    paired = A.dropna(subset=["t"]).copy()
    paired["t"] = paired["t"].astype(int)
    paired["label_a"] = m.loc[paired.a, "kl"].to_numpy()
    paired["label_t"] = m.loc[paired.t, "kl"].to_numpy()
    paired["d_primary"] = [int(np.bitwise_count(H[a] ^ H[t])) for a, t in zip(paired.a, paired.t)]
    sw = A.dropna(subset=["t_sw"]).copy()
    sw["t_sw"] = sw["t_sw"].astype(int)
    sw["label_a"] = m.loc[sw.a, "kl"].to_numpy()
    sw["label_t"] = m.loc[sw.t_sw, "kl"].to_numpy()
    sw["d_swapped"] = [int(np.bitwise_count(H[a] ^ H[t])) for a, t in zip(sw.a, sw.t_sw)]
    both = paired.merge(sw[["a", "d_swapped", "label_t"]].rename(columns={"label_t": "label_t_sw"}), on="a", how="inner")
    assert both["t_sw"].notna().all() and len(both) == len(paired), "swapped knee missing for some auto_test image"

    test_positions = np.where(folder == "test")[0]
    nearest = []
    for a in auto.index:
        d = np.bitwise_count(H[a] ^ H[test_positions]).astype(int)
        k = int(np.argmin(d))
        nearest.append((a, int(d[k]), int(test_positions[k])))
    N = pd.DataFrame(nearest, columns=["a", "d_min_test", "argmin_test"])
    other_positions = np.where(np.isin(folder, ["train", "val"]))[0]
    N["d_min_trainval"] = [int(np.bitwise_count(H[a] ^ H[other_positions]).min()) for a in N.a]
    N = N.merge(A[["a", "t"]], on="a", how="left")

    # pixel-level similarity of paired knees (Pearson correlation of grey values), primary vs swapped mapping
    def _pix(idx: int) -> np.ndarray:
        return np.asarray(Image.open(data_dir / m.loc[idx, "path"]).convert("L"), dtype=np.float64).ravel()
    ncc_p = np.array([np.corrcoef(_pix(a), _pix(t))[0, 1] for a, t in zip(both.a, both.t)])
    ncc_s = np.array([np.corrcoef(_pix(a), _pix(t))[0, 1] for a, t in zip(both.a, both.t_sw.astype(int))])
    # label-independent re-assignment: pair each auto_test image with whichever of the patient's two test knees
    # is more similar in pixels; then measure label agreement. Resolves exceptions to the k->laterality rule.
    choose_sw = ncc_s > ncc_p
    lab_chosen = np.where(choose_sw, both.label_t_sw.to_numpy(), both.label_t.to_numpy())
    prim_mismatch = (both.label_a.to_numpy() != both.label_t.to_numpy())
    reassign = {
        "images_where_other_knee_is_pixel_closer": int(choose_sw.sum()),
        "label_agreement_after_pixel_reassignment": round(float((both.label_a.to_numpy() == lab_chosen).mean()), 6),
        "label_disagreements_after_pixel_reassignment": int((both.label_a.to_numpy() != lab_chosen).sum()),
        "primary_mapping_label_mismatches": int(prim_mismatch.sum()),
        "primary_mismatches_where_other_knee_pixel_closer": int((choose_sw & prim_mismatch).sum()),
        "primary_mismatches_where_other_knee_label_agrees": int((prim_mismatch & (both.label_a.to_numpy() == both.label_t_sw.to_numpy())).sum()),
    }
    pix = {
        "pearson_primary": {"median": round(float(np.median(ncc_p)), 4), "p10": round(float(np.percentile(ncc_p, 10)), 4),
                            "min": round(float(ncc_p.min()), 4)},
        "pearson_swapped": {"median": round(float(np.median(ncc_s)), 4), "p90": round(float(np.percentile(ncc_s, 90)), 4)},
        "fraction_primary_above": {str(th): round(float((ncc_p > th).mean()), 4) for th in (0.95, 0.9, 0.8)},
        "fraction_swapped_above": {str(th): round(float((ncc_s > th).mean()), 4) for th in (0.95, 0.9, 0.8)},
    }
    phd = {f"<={th}": round(float((paired.d_primary <= th).mean()), 4) for th in (0, 4, 8, 10, 16)}

    lab_agree = float((paired.label_a == paired.label_t).mean())
    lab_agree_sw = float((sw.label_a == sw.label_t).mean())
    res["auto_test_vs_test"] = {
        "auto_test_images": int(len(auto)),
        "paired_by_patient_and_laterality": int(len(paired)),
        "unpaired": int(len(auto) - len(paired)),
        "distinct_test_knees_covered": int(paired.t.nunique()),
        "test_images_total": int(len(test)),
        "fraction_of_test_knees_covered": round(paired.t.nunique() / len(test), 6),
        "patients_covered": int(auto["pid"].nunique()),
        "label_agreement_primary_mapping": round(lab_agree, 6),
        "label_disagreements_primary_mapping": int((paired.label_a != paired.label_t).sum()),
        "label_agreement_swapped_mapping_control": round(lab_agree_sw, 6),
        "mapping_evidence_label_independent": {
            "median_phash_distance_primary": float(both.d_primary.median()),
            "median_phash_distance_swapped": float(both.d_swapped.median()),
            "fraction_primary_closer_than_swapped": round(float((both.d_primary < both.d_swapped).mean()), 6),
            "fraction_primary_le_10": round(float((paired.d_primary <= 10).mean()), 6),
            "fraction_swapped_le_10": round(float((both.d_swapped <= 10).mean()), 6),
        },
        "pixel_similarity_of_paired_knees": pix,
        "label_independent_reassignment": reassign,
        "phash_distance_of_paired_knees_cumulative_fraction": phd,
        "nearest_test_image_phash": {
            "fraction_min_distance_le_4": round(float((N.d_min_test <= 4).mean()), 6),
            "fraction_min_distance_le_8": round(float((N.d_min_test <= 8).mean()), 6),
            "fraction_min_distance_le_10": round(float((N.d_min_test <= 10).mean()), 6),
            "fraction_nearest_is_paired_knee": round(float((N.argmin_test == N.t).mean()), 6),
            "median_min_distance_to_test": float(N.d_min_test.median()),
            "median_min_distance_to_train_val": float(N.d_min_trainval.median()),
            "fraction_closer_to_test_than_to_train_val": round(float((N.d_min_test < N.d_min_trainval).mean()), 6),
        },
        "confusion_auto_rows_by_test_cols": (
            pd.crosstab(paired.label_a, paired.label_t).reindex(index=range(5), columns=range(5), fill_value=0).values.tolist()
        ),
        "auto_test_class_distribution": class_dist(auto["kl"]),
        "interpretation_limits": "Pairing lookup always succeeds because test holds both knees of each patient; "
                                 "label agreement and pixel/pHash similarity (primary vs swapped), not the lookup, support the "
                                 "k=1->R, k=2->L mapping. 'Nearest test image within pHash distance 10' is NOT specific evidence of "
                                 "duplication in this dataset (train/val images are equally close; see nearest_test_image_phash); use "
                                 "the pair-specific evidence instead. Paired knees are mostly near-identical views, not byte-identical "
                                 "(no exact SHA-256 duplicates exist). The provenance/purpose of auto_test is not documented in the "
                                 "sources accessed.",
    }
    dis = paired[paired.label_a != paired.label_t]
    dis_df = pd.DataFrame({
        "auto_test_image": m.loc[dis.a, "path"].to_numpy(),
        "test_image": m.loc[dis.t, "path"].to_numpy(),
        "auto_test_label": dis.label_a.to_numpy(), "test_label": dis.label_t.to_numpy(),
        "phash_distance": dis.d_primary.to_numpy(),
    }).sort_values("auto_test_image")
    dis_df.to_csv(out_dir / "auto_test_test_label_disagreements.csv", index=False)
    res["auto_test_vs_test"]["disagreement_label_gap_abs"] = {
        str(k): int(v) for k, v in (dis.label_a - dis.label_t).abs().value_counts().sort_index().items()
    }

    # ---- 7. bilateral L/R agreement -------------------------------------------------
    def bilateral(df: pd.DataFrame) -> dict:
        w = df.pivot_table(index="pid", columns="lat", values="kl", aggfunc="first").dropna()
        L, R = w["L"].astype(int), w["R"].astype(int)
        return {
            "patients_with_both_knees": int(len(w)),
            "exact_agreement": round(float((L == R).mean()), 6),
            "within_one_grade": round(float(((L - R).abs() <= 1).mean()), 6),
            "quadratic_weighted_kappa": round(float(cohen_kappa_score(L, R, weights="quadratic")), 6),
            "confusion_L_rows_R_cols": pd.crosstab(L, R).reindex(index=range(5), columns=range(5), fill_value=0).values.tolist(),
        }
    res["bilateral_agreement"] = {"corpus_train_val_test": bilateral(corpus)}
    role_of = corpus["path"].map(path_to_role)
    res["bilateral_agreement"]["by_splits_v1_role"] = {r: bilateral(corpus[role_of == r]) for r in ["train", "val", "test"]}

    # ---- 8. manuscript fact checks --------------------------------------------------
    c = res["counts"]; po = res["patient_overlap"]; at = res["auto_test_vs_test"]
    checks = {
        "corpus_images == 8260": c["corpus_images_train_val_test"] == 8260,
        "all_folder_images == 9786 == corpus + auto_test": c["total_images_all_folders"] == 9786 == c["corpus_images_train_val_test"] + c["auto_test_images"],
        "corpus_patients == 4130": c["patients_corpus"] == 4130,
        "exactly_two_images_per_patient (corpus)": c["images_per_patient_corpus"] == {"2": 4130},
        "0 patients in >1 of train/val/test": po["provided_folders_train_val_test"]["patients_in_more_than_one_of_train_val_test"] == 0,
        "811 patients in >1 folder, all auto_test+test": po["provided_folders_including_auto_test"]["patients_in_more_than_one_folder"] == 811
            and po["provided_folders_including_auto_test"]["folder_combinations"] == {"auto_test+test": 811},
        "auto_test images == 1526": c["auto_test_images"] == 1526,
        "splits_v1 patient-disjoint": po["splits_v1"]["patients_in_multiple_roles"] == 0,
        "no exact (SHA-256) duplicates": res["exact_duplicates"]["duplicate_groups"] == 0,
        "manifest sha256 matches files": res["exact_duplicates"]["manifest_sha256_matches_recomputed"],
        "auto_test label agreement (primary) > 0.98 and swapped control < 0.60": at["label_agreement_primary_mapping"] > 0.98
            and at["label_agreement_swapped_mapping_control"] < 0.60,
        "pixel-based re-assignment gives label agreement >= 0.99": at["label_independent_reassignment"]["label_agreement_after_pixel_reassignment"] >= 0.99,
        "paired-knee pixel r: primary median > 0.9 and swapped median < 0.8": at["pixel_similarity_of_paired_knees"]["pearson_primary"]["median"] > 0.9
            and at["pixel_similarity_of_paired_knees"]["pearson_swapped"]["median"] < 0.8,
    }
    res["manuscript_fact_checks"] = {k: bool(v) for k, v in checks.items()}
    (out_dir / "dataset_integrity_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    # ---- human-readable companion ---------------------------------------------------
    write_md(res, out_dir / "dataset_integrity_audit.md")
    print(json.dumps(res["manuscript_fact_checks"], indent=2))
    print("wrote", out_dir / "dataset_integrity_audit.json")


def _dist_row(name: str, d: dict) -> str:
    tot = sum(d.values())
    return f"| {name} | " + " | ".join(str(d[str(k)]) for k in range(5)) + f" | {tot} |"


def write_md(r: dict, path: Path) -> None:
    c, po, ex, nd, at, bl = (r["counts"], r["patient_overlap"], r["exact_duplicates"],
                             r["near_duplicates_phash"], r["auto_test_vs_test"], r["bilateral_agreement"])
    cd = r["class_distributions"]
    L = []
    L.append("# Dataset Integrity Audit (auto-generated)\n")
    L.append("**Generated by `scripts/dataset_integrity_audit.py`. Do not edit by hand; rerun the script.** "
             "Machine-readable numbers: `dataset_integrity_audit.json`.\n")
    m = r["meta"]
    L.append(f"- Generated: {m['generated_at']} | Python {m['python']} | numpy {m['numpy']} | pandas {m['pandas']}")
    L.append(f"- Git HEAD: `{m['git']['head']}` (working tree dirty: {m['git']['dirty']})")
    for k, v in m["inputs"].items():
        L.append(f"- Input `{v['path']}` sha256 `{v['sha256'][:16]}…`")
    L.append("- pHash: 64-bit DCT pHash, same definition as `scripts/forensic_audit.py`.\n")

    L.append("## Manuscript fact checks\n")
    L.append("| Check | Result |\n|---|---|")
    for k, v in r["manuscript_fact_checks"].items():
        L.append(f"| {k} | {'PASS' if v else '**FAIL**'} |")

    L.append("\n## 1. Counts\n")
    L.append(f"- Images, all four folders: **{c['total_images_all_folders']}** ({c['images_per_provided_folder']})")
    L.append(f"- Corpus (train+val+test): **{c['corpus_images_train_val_test']}** images; `auto_test`: **{c['auto_test_images']}**; "
             f"all − corpus = {c['total_minus_corpus']}")
    L.append(f"- Patients (parsed 7-digit ids): all folders **{c['patients_all_folders']}**, corpus **{c['patients_corpus']}**, `auto_test` **{c['patients_auto_test']}**")
    L.append(f"- Patients per provided folder: {c['patients_per_provided_folder']}")
    L.append(f"- Images per patient in corpus (images:patients): {c['images_per_patient_corpus']}; unique (patient, laterality) knees: {c['knees_unique_patient_lat_corpus']}")
    L.append(f"- Image dimensions: {c['image_dimensions']}; unparsed file names: {c['unparsed_file_names']}\n")

    L.append("## 2. Class distributions (KL 0–4)\n")
    L.append("| Set | KL0 | KL1 | KL2 | KL3 | KL4 | Total |\n|---|---|---|---|---|---|---|")
    for k, v in cd["by_provided_folder"].items():
        L.append(_dist_row(f"provided `{k}`", v))
    L.append(_dist_row("corpus (train+val+test)", cd["corpus_train_val_test"]))
    L.append(_dist_row("all four folders", cd["all_folders_incl_auto_test"]))
    for k, v in cd["splits_v1_by_role"].items():
        L.append(_dist_row(f"`splits_v1` {k}", v))

    L.append("\n## 3. Patient overlap\n")
    a = po["provided_folders_train_val_test"]
    L.append(f"- **Provided train/val/test:** patients in more than one of the three folders = **{a['patients_in_more_than_one_of_train_val_test']}** of {a['patients']}")
    b = po["provided_folders_including_auto_test"]
    L.append(f"- **Including `auto_test`:** patients in more than one folder = **{b['patients_in_more_than_one_folder']}** ({b['fraction_of_all_patients']:.4f} of all patients); combinations: {b['folder_combinations']}")
    L.append(f"- `auto_test` patients not in `test`: {po['auto_test_patients_not_in_test']}; `test` patients not in `auto_test`: {po['test_patients_not_in_auto_test']}")
    s = po["splits_v1"]
    L.append(f"- **`splits_v1`:** patients {s['patients']}; in multiple roles **{s['patients_in_multiple_roles']}**; images per role {s['images_per_role']}; patients per role {s['patients_per_role']}\n")

    L.append("## 4. Exact duplicates (SHA-256 recomputed from files)\n")
    L.append(f"- Duplicate groups: **{ex['duplicate_groups']}**; images in duplicate groups: {ex['images_in_duplicate_groups']}; manifest hashes match files: {ex['manifest_sha256_matches_recomputed']}\n")

    L.append("## 5. Near-duplicates (64-bit DCT pHash, Hamming distance)\n")
    L.append("Pairs counted once (i<j). Different-patient pairs at small distances are mostly structurally similar knees, not duplicates.\n")
    for title, key in [("All four folders, by provided-folder pair", "all_folders_by_provided_folder_pair"),
                       ("Corpus only, by provided-folder pair", "corpus_only_by_provided_folder_pair"),
                       ("Corpus only, by `splits_v1` role pair", "corpus_only_by_splits_v1_role_pair")]:
        L.append(f"### {title}\n")
        L.append("| Threshold | Pair type | Same patient | Different patient |\n|---|---|---|---|")
        for thr, v in nd[key].items():
            L.append(f"| {thr} | **all** | {v['same_patient']} | {v['different_patient']} |")
            for pt, x in v["by_pair_type"].items():
                L.append(f"| {thr} | {pt} | {x['same_patient']} | {x['different_patient']} |")
        L.append("")
    L.append(f"- Corpus pairs with identical pHash (d=0): {nd['corpus_identical_phash_pairs_d0']}\n")

    L.append("## 6. `auto_test` ↔ `test` pairing\n")
    L.append(f"- `auto_test` images: {at['auto_test_images']}; paired by patient id + laterality (`_1`→R, `_2`→L, inferred): **{at['paired_by_patient_and_laterality']}**; unpaired: {at['unpaired']}")
    L.append(f"- Distinct `test` knees covered: **{at['distinct_test_knees_covered']}** of {at['test_images_total']} ({at['fraction_of_test_knees_covered']:.4f}); patients covered: {at['patients_covered']}")
    L.append(f"- **Label agreement (primary mapping): {at['label_agreement_primary_mapping']:.4f}** ({at['label_disagreements_primary_mapping']} disagreements; abs. label gap {at['disagreement_label_gap_abs']}); "
             f"control with swapped mapping: {at['label_agreement_swapped_mapping_control']:.4f}")
    e = at["mapping_evidence_label_independent"]
    L.append(f"- Label-independent mapping evidence: median pHash distance primary {e['median_phash_distance_primary']} vs swapped {e['median_phash_distance_swapped']}; "
             f"primary closer than swapped in {e['fraction_primary_closer_than_swapped']:.4f}; ≤10 primary {e['fraction_primary_le_10']:.4f} vs swapped {e['fraction_swapped_le_10']:.4f}")
    px = at["pixel_similarity_of_paired_knees"]
    L.append(f"- **Pixel similarity of paired knees (Pearson r of grey values):** primary mapping median **{px['pearson_primary']['median']}** "
             f"(10th percentile {px['pearson_primary']['p10']}, min {px['pearson_primary']['min']}) vs swapped mapping median {px['pearson_swapped']['median']}; "
             f"fraction with r>0.95 / >0.9 / >0.8: primary {px['fraction_primary_above']} vs swapped {px['fraction_swapped_above']}")
    ra = at["label_independent_reassignment"]
    L.append(f"- **Exceptions to the `_1`→R / `_2`→L rule:** {ra['primary_mapping_label_mismatches']} images mismatch under the rule; "
             f"in {ra['primary_mismatches_where_other_knee_pixel_closer']} of them the patient's *other* knee is pixel-closer and in "
             f"{ra['primary_mismatches_where_other_knee_label_agrees']} the other knee's label agrees. Pairing each `auto_test` image with the "
             f"pixel-closer of the patient's two `test` knees (label-independent) gives label agreement **{ra['label_agreement_after_pixel_reassignment']:.4f}** "
             f"({ra['label_disagreements_after_pixel_reassignment']} disagreements; {ra['images_where_other_knee_is_pixel_closer']} images re-assigned).")
    L.append(f"- pHash distance of paired knees, cumulative fraction: {at['phash_distance_of_paired_knees_cumulative_fraction']} "
             "(moderate distances: the paired images are near-identical views, not pixel-identical files)")
    L.append("- **Caution:** the statistic 'nearest `test` image within pHash distance ≤10' (see next bullet) is *not* duplicate evidence here: "
             "the nearest `test` image is the paired knee in only about a third of cases and train/val images are equally close.")
    q = at["nearest_test_image_phash"]
    L.append(f"- Nearest `test` image by pHash: min distance ≤4 in {q['fraction_min_distance_le_4']:.4f}, ≤8 in {q['fraction_min_distance_le_8']:.4f}, ≤10 in {q['fraction_min_distance_le_10']:.4f}; "
             f"nearest is the paired knee in {q['fraction_nearest_is_paired_knee']:.4f}; closer to `test` than to train/val in {q['fraction_closer_to_test_than_to_train_val']:.4f} "
             f"(median min distance test {q['median_min_distance_to_test']} vs train/val {q['median_min_distance_to_train_val']})")
    L.append(f"- Limits: {at['interpretation_limits']}")
    L.append("- Disagreeing pairs are listed in `auto_test_test_label_disagreements.csv`.\n")

    L.append("## 7. Bilateral (left/right knee) label agreement\n")
    x = bl["corpus_train_val_test"]
    L.append(f"- Corpus: {x['patients_with_both_knees']} patients; exact agreement **{x['exact_agreement']:.4f}**; within one grade **{x['within_one_grade']:.4f}**; quadratic-weighted κ **{x['quadratic_weighted_kappa']:.4f}**")
    for rr, y in bl["by_splits_v1_role"].items():
        L.append(f"  - `splits_v1` {rr}: n={y['patients_with_both_knees']}, exact {y['exact_agreement']:.4f}, ≤1 grade {y['within_one_grade']:.4f}, QWK {y['quadratic_weighted_kappa']:.4f}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
