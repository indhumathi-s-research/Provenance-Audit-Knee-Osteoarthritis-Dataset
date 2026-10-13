"""Run multiple pilot configs sequentially, logging each to runs/<id>_train.log.

Usage:
    python scripts/run_pilots_sequentially.py configs/m2.yaml configs/m3.yaml ...
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    configs = [Path(c) for c in sys.argv[1:]]
    if not configs:
        print("usage: python scripts/run_pilots_sequentially.py configs/mX.yaml ...")
        sys.exit(1)

    for cfg in configs:
        exp_id = cfg.stem.upper()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = PROJECT_ROOT / "runs" / f"{exp_id}_{stamp}_train.log"
        err = PROJECT_ROOT / "runs" / f"{exp_id}_{stamp}_train.err.log"
        print(f"[{time.strftime('%H:%M:%S')}] launching {cfg} ...", flush=True)
        t0 = time.time()
        with open(log, "w") as fout, open(err, "w") as ferr:
            rc = subprocess.run(
                [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "train_pilot.py"),
                 "--config", str(cfg)],
                cwd=str(PROJECT_ROOT), stdout=fout, stderr=ferr,
            )
        print(f"[{time.strftime('%H:%M:%S')}] {exp_id} finished in "
              f"{(time.time()-t0)/60:.1f} min (rc={rc.returncode}) | log: {log}", flush=True)
        if rc.returncode != 0:
            print(f"  ERROR tail: {err.read_text(errors='replace')[-800:]}", flush=True)


if __name__ == "__main__":
    main()
