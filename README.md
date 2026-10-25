# IMAD — Multimodal Fake Content Detector

Research software for assessing whether image, text, or naturally paired image-text content resembles examples in labeled datasets. Outputs are model scores, not proof of authenticity. False positives, distribution shift, and dataset artifacts are expected risks.

## Current state

The project has a GenImage-specific dataset parser, metadata/sample/full audit modes, duplicate-aware split manifests, tests, and a ResNet-18 transfer-learning workflow that saves validation-selected checkpoints and internal-test reports. No dataset or weights are included. The local bundled Python runtime currently has no PyTorch or pytest, and no real experiment has been trained or evaluated.

## Quick start

Use Python 3.10+ in a virtual environment. For image workflows install `pip install -e ".[image,dev]"`. The optional groups separate CPU-friendly API and data tooling from heavyweight ML packages. Verify the selected PyTorch build against the machine and GPU before installing.

```powershell
python scripts/check_environment.py
python scripts/prepare_genimage.py <extracted-genimage-root> --metadata-only --inspection-output reports/experiments/genimage-metadata.json
python scripts/prepare_genimage.py <extracted-genimage-root> --sample-size 32 --inspection-output reports/experiments/genimage-sample.json
python scripts/prepare_genimage.py <extracted-genimage-root> --seed 42 --manifest-output data/manifests/genimage.csv --inspection-output reports/experiments/genimage-audit.json
python scripts/train_genimage_baseline.py data/manifests/genimage.csv --output-dir models/genimage_resnet18 --epochs 5
pytest
```

GenImage is manually acquired from the [official source page](https://github.com/GenImage-Dataset/GenImage); its README links Google Drive and Baidu archives (Baidu code `ztf1`). Download only a selected generator archive for the first experiment and preserve `<generator>/{train,val}/{ai,nature}/...`. The data terms require noncommercial academic/research use under CC BY-NC-SA 4.0 plus additional terms. No downloader is provided, and no data is downloaded automatically. Do not commit extracted images.

The GenImage-specific parser requires this documented layout and maps `ai → fake`, `nature → real` (ImageNet-collected). Metadata-only mode counts files without decoding. Sample mode verifies a deterministic split/class-stratified subset and cannot create a training manifest. Full manifest generation decodes and hashes all files; it preserves official test if present. Otherwise, with multiple compatible generator folders it holds one complete generator out as internal test; for a single-generator subset it keeps official validation and carves internal test from official train. SHA-256 and same-class equal-dHash duplicates are grouped across splits; exact duplicates with conflicting labels and duplicate groups crossing official splits are rejected. dHash is a practical cue, not comprehensive semantic duplicate detection. Keep `source_split`/`generator` metadata and review manifest counts before training.

The trainer selects a checkpoint by validation loss, then evaluates the internal test only after a full run. It uses class-weighted loss and explicitly maps the fake-score column by label. It saves configuration, label mapping, train history, best/latest checkpoints, test metrics and predictions, and plots. Use `--smoke-test` only to validate training/validation and checkpoint I/O; it never opens the internal test images or emits test metrics.

`evaluate_predictions.py` accepts `y_true,y_pred` columns and optional `positive_score` values. Supply the exact positive label from the dataset. ROC-AUC is computed only for binary truth labels; score semantics are reported as uncalibrated.

## Data provenance and roadmap

See [dataset protocol](docs/dataset_protocol.md), [methodology](docs/methodology.md), [architecture](docs/architecture.md), and [limitations](docs/limitations.md). Dataset download is intentionally manual and dataset-specific. Review each source's current license, task labels, split definitions, and access terms before use. Never place downloaded data in Git.

Phases 1–3 remain in progress. The target datasets are GenImage, DiffusionForensics/DIR, NIST OpenMFC, M4/SemEval-2024 Task 8, RAID, and OmniFake98K. External datasets remain held out from model selection. Phase 3 requires genuine pairs from OmniFake98K; independent image/text data must never be combined to simulate pairs.

## Colab

Push the reviewed repository changes, open `notebooks/phase1_image_baseline.ipynb` in Colab, select a GPU runtime, set `DATA_ROOT` to the extracted archive, and run metadata/sample checks, full manifest generation, smoke fit, then full training. Save artifacts to Drive. Notebook runtime execution is not verified from this local environment.

## Development

Install `pip install -e ".[dev]"`; run `pytest`. See [API documentation](docs/api.md) for planned service contracts. Model weights are trusted local artifacts; do not load arbitrary checkpoints from untrusted sources.
