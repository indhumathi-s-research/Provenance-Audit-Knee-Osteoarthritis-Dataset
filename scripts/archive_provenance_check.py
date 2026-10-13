"""Record the provenance of the audited dataset copy from the locally downloaded Kaggle archive (read-only).

Compares a downloaded archive with the audited manifest and records archive-level metadata:
  - size, SHA-256, file-system creation/modification time
  - Windows "Zone.Identifier" download metadata, reduced to: referrer host, the URL PATH of the download
    (without its query string, which contains a temporary signed credential) and the signing date
  - entry counts per top-level folder
  - whether every archive entry matches a manifest path and has an identical SHA-256

It does not read browser history. It does not identify the Kaggle slug (the archive metadata does not contain it);
the slug is an author statement recorded separately in the evidence registry.

Usage:
    python scripts/archive_provenance_check.py --archive C:/Users/<user>/Downloads/data-knee.zip
Output: research/results/dataset_archive_provenance.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "results" / "dataset_archive_provenance.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def zone_identifier(path: Path) -> dict | None:
    try:
        with open(str(path) + ":Zone.Identifier", "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return None
    kv = dict(line.split("=", 1) for line in txt.splitlines() if "=" in line)
    out = {"ZoneId": kv.get("ZoneId"), "ReferrerUrl": kv.get("ReferrerUrl")}
    host = kv.get("HostUrl")
    if host:
        u = urlparse(host)
        out["host"] = u.netloc
        out["download_url_path_without_query"] = u.path
        m = re.search(r"X-Goog-Date=(\d{8}T\d{6}Z)", host)
        out["signing_date_utc"] = m.group(1) if m else None
    return out


def git_state() -> dict:
    def run(*a):
        r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"head": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "master.csv"))
    a = ap.parse_args()
    arc = Path(a.archive)
    st = arc.stat()
    zf = zipfile.ZipFile(arc)
    names = [n for n in zf.namelist() if not n.endswith("/")]

    m = pd.read_csv(a.manifest)
    man = {p.replace("\\", "/"): h for p, h in zip(m.image_path, m.sha256)}

    def key(n: str):
        parts = n.split("/")
        for i in range(len(parts)):
            k = "/".join(parts[i:])
            if k in man:
                return k
        return None

    matched = identical = 0
    seen, unmatched = set(), []
    for n in names:
        k = key(n)
        if k is None:
            unmatched.append(n)
            continue
        matched += 1
        seen.add(k)
        if hashlib.sha256(zf.read(n)).hexdigest() == man[k]:
            identical += 1

    res = {
        "meta": {"script": "scripts/archive_provenance_check.py", "git": git_state(),
                 "generated_at": datetime.datetime.now().isoformat(timespec="seconds")},
        "archive": {
            "file_name": arc.name,
            "size_bytes": st.st_size,
            "sha256": sha256_file(arc),
            "fs_creation_time": datetime.datetime.fromtimestamp(st.st_ctime).astimezone().isoformat(timespec="seconds"),
            "fs_modification_time": datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
            "zone_identifier": zone_identifier(arc),
        },
        "contents": {
            "entries": len(names),
            "top_level_folders": dict(Counter(n.split("/")[0] for n in names)),
            "matched_to_manifest_paths": matched,
            "sha256_identical_to_manifest": identical,
            "unmatched_archive_entries": len(unmatched),
            "manifest_files_missing_from_archive": len(set(man) - seen),
        },
        "notes": [
            "The archive metadata identifies kaggle.com as referrer and a Kaggle storage path; it does not contain the dataset slug.",
            "The Kaggle slug is an author statement (shashwatwork/knee-osteoarthritis-dataset-with-severity), recorded in the evidence registry.",
            "The signed download URL query string is intentionally not stored.",
        ],
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("archive", "contents")}, indent=2))


if __name__ == "__main__":
    main()
