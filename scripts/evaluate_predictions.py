"""Evaluate a CSV containing y_true, y_pred, and optionally positive_score."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imad.evaluation.metrics import classification_metrics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("predictions", type=Path)
parser.add_argument("--positive-label", required=True, help="Exact dataset label treated as positive")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
with args.predictions.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows or not {"y_true", "y_pred"}.issubset(rows[0]):
    parser.error("CSV needs y_true and y_pred columns")
scores = [float(r["positive_score"]) for r in rows] if rows[0].get("positive_score") else None
report = classification_metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows],
    positive_label=args.positive_label, positive_scores=scores)
encoded = json.dumps(report, indent=2)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
print(encoded)
