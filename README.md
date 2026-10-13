# Provenance Audit of a Knee-Osteoarthritis KL-Grading Dataset

Code accompanying the paper **"A Data-Integrity and Provenance Audit of a Public Knee-Osteoarthritis KL-Grading Dataset"**.

The repository contains the scripts that:

1. audit the Kaggle redistribution *Knee Osteoarthritis Dataset with Severity Grading* (folders `train`, `val`, `test`, `auto_test`), derived from Chen (2018), Mendeley Data, [doi:10.17632/56rmx5bjcr.1](https://doi.org/10.17632/56rmx5bjcr.1);
2. regenerate every dataset statistic reported in the paper (image and patient counts, class distributions, patient overlap between folders, `auto_test` ↔ `test` pairing, exact and near-duplicate statistics, label agreement);
3. train and evaluate the illustrative single-run baselines (M1–M6) and compute their provenance statistics.

**No images, per-image manifests or patient identifiers are distributed here.** Obtain the original images from the source distribution under its terms.

## Repository layout

```
configs/    YAML configurations of the six baselines (M1–M6)
scripts/    Command-line entry points (audit, split, training, reporting)
src/        Shared library code (metadata parsing, audit, splits, models, metrics, tracking)
```

## Installation

Python 3.11 was used. A CUDA GPU is needed only for the training scripts; the dataset audit runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run all commands from the repository root.

## Data

Download the Kaggle archive and extract it so that the images sit at:

```
data/train/{0..4}/*.png
data/val/{0..4}/*.png
data/test/{0..4}/*.png
data/auto_test/{0..4}/*.png
```

The class directory name (0–4) is the KL grade. The seven-digit file-name prefix is parsed as the patient identifier. `data/` is excluded from version control.

## Reproducing the dataset audit

| Step | Command | Main outputs |
|---|---|---|
| 1. Manifest and forensic audit | `python scripts/forensic_audit.py --data-dir data` | `data/manifests/master.csv`, `data/reports/*` |
| 2. Dataset metadata and version hash | `python scripts/inspect_dataset.py --data-dir data` | `data/processed/dataset_metadata.csv`, `data/processed/dataset_version.txt` |
| 3. Patient-wise 70/15/15 split (seed 42) | `python scripts/split_dataset.py` | `data/processed/splits_v1.csv`, `data/reports/split_audit.md` |
| 4. Freeze the corpus | `python scripts/freeze_dataset.py` | `data/processed/KOA_KL_PILOT_v1_freeze.json` |
| 5. Integrity audit (all paper statistics) | `python scripts/dataset_integrity_audit.py --data-dir data --out-dir research/results` | `dataset_integrity_audit.json`, `dataset_integrity_audit.md`, `auto_test_test_label_disagreements.csv` |
| 6. Archive provenance (optional) | `python scripts/archive_provenance_check.py --archive <path-to-downloaded.zip>` | `research/results/dataset_archive_provenance.json` |

Step 5 is read-only on `data/`. Its outputs record the git commit and the SHA-256 of their inputs.

Step 6 compares the downloaded archive with the manifest entry by entry (SHA-256). On Windows it also reads the archive's `Zone.Identifier` download metadata, keeping only the referrer host and URL path. It never stores the signed query string.

## Reproducing the baselines

| ID | Model | Resolution | Preprocessing |
|---|---|---|---|
| M1 | Light CNN | 224×224 | none |
| M2 | Standard CNN | 224×224 | none |
| M3 | Standard CNN | 300×300 | none |
| M4 | Standard CNN | 224×224 | CLAHE |
| M5 | Standard CNN | 224×224 | ROI centre crop |
| M6 | ResNet-18 (ImageNet-pretrained) | 224×224 | CLAHE + ImageNet normalisation |

All baselines use weighted cross-entropy, with class weights computed from the training split only. Model selection uses validation macro-F1. The test set is evaluated once, for the selected configuration only.

```bash
# Train each configuration on validation (test never loaded)
python scripts/run_baseline.py --config configs/m1.yaml   # repeat for m2 ... m6

# Select on validation macro-F1, evaluate the selected model on test once
python scripts/select_and_eval_test.py

# Comparison tables and figures -> reports/
python scripts/baseline_report.py

# Standalone latency / FLOPs / memory measurement
python scripts/measure_compute.py --config configs/m6.yaml --image <path-to-one-test-png>
```

Outputs go to `experiments/baseline/<ID>/`: config, validation metrics, predictions, training history, compute metrics and run metadata (environment, git commit, dataset and split hashes).

### Baseline provenance statistics

The paper compares two baseline generations: generation A, from an earlier pilot pipeline, and generation B, from the frozen-corpus pipeline above.

```bash
# Generation A (earlier pilot pipeline)
python scripts/train_pilot.py --config configs/m6.yaml --runs-dir runs
python scripts/run_pilots_sequentially.py configs/m1.yaml configs/m2.yaml ...   # several configs
python scripts/pilot_report.py

# Statistics across generations A and B, including the patient-cluster bootstrap
python scripts/baseline_provenance_stats.py
```

`baseline_provenance_stats.py` only reads stored predictions and metrics; it does not train anything. It writes `research/results/baseline_provenance_stats.json`.

### Hyperparameter search (exploratory)

`scripts/optimize.py --method {random,pso,soa}` runs a hyperparameter search over M6's search space, optimising validation macro-F1. In the reported work only random search completed (20 evaluations). Its update rule is not the published Seagull Optimization Algorithm. The paper does not draw conclusions from this search.

## Notes

- Some script docstrings and log messages date from the pilot phase, and a few statements in them were revised later. For example, `auto_test` is not a re-encoded copy of `test`: it holds one automatically detected crop per test knee, and the distributed `train`/`val`/`test` folders are patient-disjoint by file-name identifier. Where a docstring and the paper disagree, the paper and the output of `scripts/dataset_integrity_audit.py` are authoritative.
- Runs record the git commit of the repository they ran in. Commit hashes in the paper's stored outputs refer to the authors' working repository, not to this public release.
- Hardware used for the reported runs: NVIDIA GTX 1050 Ti (4 GB), FP32.

## Data use and licence

- Underlying dataset: Chen, P. (2018). *Knee Osteoarthritis Severity Grading Dataset* (Version 1). Mendeley Data. https://doi.org/10.17632/56rmx5bjcr.1 (listed under CC BY 4.0). The images originate from the Osteoarthritis Initiative (OAI).
- This repository's code is released under the MIT License (see `LICENSE`). The licence covers the code only, not the dataset.

## Citation

If you use this code, please cite the paper (citation details will be added on publication) and the archived release of this repository (DOI to be added).
