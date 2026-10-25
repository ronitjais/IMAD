"""Inspect an image dataset without downloading or modifying it."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imad.data.images import inspect_images

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path, help="Root directory of an already available dataset")
parser.add_argument("--sample-size", type=int, default=None, help="Inspect a reproducible sample only")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, help="Optional JSON report path")
args = parser.parse_args()
report = inspect_images(args.dataset, args.sample_size, args.seed)
encoded = json.dumps(report, indent=2)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
print(encoded)
