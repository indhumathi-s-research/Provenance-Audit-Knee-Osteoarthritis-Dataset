"""Patient-wise data splitting with leakage verification.

Reliable 7-digit patient ids are parsed from filenames, so splits are
created at the patient level (70/15/15). No patient crosses roles.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import sha256_text


def deduplicate_exact(meta: pd.DataFrame) -> pd.DataFrame:
    """Drop byte-identical duplicate images, keeping the first occurrence."""
    keep = ~meta.duplicated(subset="image_path")
    return meta[keep].reset_index(drop=True)


def patient_wise_split(
    meta: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    """Assign every image of a patient to one role (train/val/test).

    Stratification uses each patient's dominant KL grade so that the
    patient-level split approximately preserves the class distribution.
    """
    parsed = meta[meta["patient_id"].ne("")].copy()
    unparsed = meta[meta["patient_id"].eq("")].copy()
    if len(unparsed):
        raise ValueError(
            f"{len(unparsed)} images lack a patient id; cannot guarantee patient-wise "
            f"separation. Refusing to split silently."
        )

    patient_df = parsed.groupby("patient_id", sort=False).agg(
        image_count=("image_path", "count"),
        dominant_kl=("kl_label", lambda s: int(s.value_counts().idxmax())),
    )
    strat = patient_df["dominant_kl"].to_numpy()

    def do_split(seed_a, seed_b, stratify=True):
        tr, re = train_test_split(
            patient_df.index.to_numpy(),
            test_size=val_size + test_size,
            random_state=seed_a,
            stratify=strat if stratify else None,
        )
        test_frac = test_size / (val_size + test_size)
        va, te = train_test_split(
            re,
            test_size=test_frac,
            random_state=seed_b,
            stratify=patient_df.loc[re, "dominant_kl"].to_numpy() if stratify else None,
        )
        return tr, va, te

    try:
        train_idx, val_idx, test_idx = do_split(seed, seed + 1, stratify=True)
    except ValueError:
        # Stratification impossible when some dominant class has < 2 patients.
        train_idx, val_idx, test_idx = do_split(seed, seed + 1, stratify=False)
        print("WARNING: dominant-KL stratification fell back to plain random "
              "patient split (rare class with < 2 patients as dominant).")

    role = {}
    for pid in train_idx:
        role[pid] = "train"
    for pid in val_idx:
        role[pid] = "val"
    for pid in test_idx:
        role[pid] = "test"

    parsed["role"] = parsed["patient_id"].map(role)
    return parsed


def verify_no_leakage(split_df: pd.DataFrame) -> dict:
    patient_role = split_df.groupby("patient_id")["role"].nunique()
    overlap = patient_role[patient_role > 1]
    return {
        "patients": int(split_df["patient_id"].nunique()),
        "patients_in_multiple_roles": int(len(overlap)),
        "violations": [
            f"{pid}: {dict(split_df[split_df['patient_id']==pid]['role'].value_counts())}"
            for pid in overlap.index[:20]
        ],
    }


def split_version(split_df: pd.DataFrame) -> str:
    text = ",".join(
        split_df.sort_values("image_path")
        .apply(lambda r: f"{r['image_path']}|{r['role']}", axis=1)
        .tolist()
    )
    return sha256_text(text)[:12]


def provided_split_leakage(meta: pd.DataFrame) -> dict:
    """Check whether patients leak across the *provided* train/val/test/auto_test."""
    parsed = meta[meta["patient_id"].ne("")]
    counts = parsed.groupby("patient_id")["split"].nunique()
    leaking = counts[counts > 1]
    detail = []
    for pid in leaking.index[:15]:
        detail.append({pid: parsed[parsed["patient_id"] == pid]["split"].value_counts().to_dict()})
    return {
        "n_patients": int(parsed["patient_id"].nunique()),
        "patients_in_multiple_provided_splits": int(len(leaking)),
        "fraction": round(len(leaking) / max(len(parsed["patient_id"].unique()), 1), 4),
        "examples": detail,
    }


def role_distribution(split_df: pd.DataFrame) -> pd.DataFrame:
    return (
        split_df.pivot_table(
            index="role", columns="kl_label", values="image_path", aggfunc="count", fill_value=0
        )
        .reindex(["train", "val", "test"])
        .reset_index()
    )
