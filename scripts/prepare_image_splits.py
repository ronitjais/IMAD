"""Inspect images and write a seeded, exact-duplicate-safe CSV split manifest."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imad.data.images import create_split_manifest, inspect_images

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validation-fraction", type=float, default=0.15)
parser.add_argument("--test-fraction", type=float, default=0.15)
args = parser.parse_args()
report = inspect_images(args.dataset)
splits = create_split_manifest(report["records"], seed=args.seed,
    validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "source_hint", "source_split_hint"])
    writer.writeheader()
    writer.writerows(splits)
print(f"Wrote {len(splits)} records to {args.output}. Review inferred labels and class coverage before training.")
