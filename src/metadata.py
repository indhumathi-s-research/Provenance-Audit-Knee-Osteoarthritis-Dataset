"""Parses raw dataset folder structure into a metadata DataFrame."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PATIENT_LAT = re.compile(r"^(?P<pid>\d+)(?P<laterality>[LlRr])\.(?P<ext>[a-zA-Z]+)$")
PATIENT_VIEW = re.compile(r"^(?P<pid>\d+)_(?P<view>\d+)\.(?P<ext>[a-zA-Z]+)$")


def parse_filename(stem: str) -> dict:
    """Extract patient id, laterality / view token and extension from a filename.

    Returns parsed fields or empty dict when nothing recognizable is present.
    """
    out = {}
    m = PATIENT_LAT.match(stem)
    if m:
        out["patient_id"] = m.group("pid")
        out["laterality"] = m.group("laterality").upper()
        out["view_token"] = ""
        return out
    m = PATIENT_VIEW.match(stem)
    if m:
        out["patient_id"] = m.group("pid")
        out["laterality"] = ""
        out["view_token"] = m.group("view")
        return out
    m = re.match(r"^(?P<pid>\d+)\.(?P<ext>[a-zA-Z]+)$", stem)
    if m:
        out["patient_id"] = m.group("pid")
        out["laterality"] = ""
        out["view_token"] = ""
        return out
    return out


def build_metadata(data_root: Path) -> pd.DataFrame:
    """Scan data_root/{split}/{kl_class}/*.png into a metadata DataFrame.

    Each row: image_path, split, kl_label, file_name, stem, ext,
    patient_id, laterality, view_token, parsed (bool).
    """
    data_root = Path(data_root)
    rows = []
    splits = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    for split in splits:
        split_dir = data_root / split
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            if not re.match(r"^\d+$", class_dir.name):
                # non-numeric folder (not a KL class) — record but flag
                label = None
            else:
                label = int(class_dir.name)
            for img_path in sorted(class_dir.iterdir()):
                if not img_path.is_file():
                    continue
                ext = img_path.suffix.lower()
                if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                    continue
                parsed = parse_filename(img_path.name)
                rows.append(
                    {
                        "image_path": str(img_path),
                        "split": split,
                        "kl_label": label,
                        "file_name": img_path.name,
                        "stem": img_path.stem,
                        "ext": ext,
                        "patient_id": parsed.get("patient_id", ""),
                        "laterality": parsed.get("laterality", ""),
                        "view_token": parsed.get("view_token", ""),
                        "parsed": bool(parsed),
                    }
                )
    df = pd.DataFrame(rows)
    df["knee_id"] = df.apply(
        lambda r: f"{r['patient_id']}{r['laterality']}"
        if r["patient_id"] and r["laterality"]
        else (f"{r['patient_id']}_{r['view_token']}" if r["patient_id"] else ""),
        axis=1,
    )
    df = df.sort_values(["split", "kl_label", "file_name"]).reset_index(drop=True)
    return df
