"""Inspect a manually downloaded GenImage tree and optionally prepare a manifest.

Never downloads files. Full manifest preparation decodes and hashes all images and
can take substantial time; metadata-only/sample modes are safe for initial checks.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imad.data.genimage import create_genimage_manifest, inspect_genimage

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path, help="Root containing extracted generator directories")
parser.add_argument("--metadata-only", action="store_true", help="Count paths without opening image files")
parser.add_argument("--sample-size", type=int, help="Audit only a deterministic sample; cannot produce a manifest")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--internal-test-fraction", type=float, default=0.15)
parser.add_argument("--inspection-output", type=Path)
parser.add_argument("--manifest-output", type=Path)
args = parser.parse_args()
if args.manifest_output and (args.metadata_only or args.sample_size is not None):
    parser.error("A complete manifest requires a full image audit; remove --metadata-only/--sample-size")
report = inspect_genimage(args.dataset, sample_size=args.sample_size, seed=args.seed,
                          metadata_only=args.metadata_only)
if args.inspection_output:
    args.inspection_output.parent.mkdir(parents=True, exist_ok=True)
    saved_report = {key: value for key, value in report.items() if key != "records"}
    args.inspection_output.write_text(json.dumps(saved_report, indent=2) + "\n", encoding="utf-8")
if args.manifest_output:
    manifest = create_genimage_manifest(report["records"], seed=args.seed,
                                        internal_test_fraction=args.internal_test_fraction)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    test_generators = {row["generator"] for row in manifest if row["split"] == "internal_test"}
    other_generators = {row["generator"] for row in manifest if row["split"] != "internal_test"}
    if any(row["source_split"] == "internal_test" for row in manifest):
        strategy = "official_test_preserved"
    elif not test_generators.intersection(other_generators):
        strategy = "generator_source_holdout"
    else:
        strategy = "seeded_hash_group_holdout_from_official_train"
    metadata = {
        "dataset": "GenImage", "source_page": "https://github.com/GenImage-Dataset/GenImage",
        "dataset_root": str(args.dataset.expanduser().resolve()), "seed": args.seed,
        "internal_test_fraction": args.internal_test_fraction,
        "split_strategy": strategy, "label_mapping": {"ai": "fake", "nature": "real"},
        "discovered_records": report["total_discovered"], "audited_records": report["audited"],
        "corrupt_records_excluded": report["corrupt_count"],
        "duplicate_records_removed": report["audited"] - len(manifest),
        "split_counts": {split: sum(row["split"] == split for row in manifest)
                         for split in ("train", "validation", "internal_test")},
        "manifest_sha256": hashlib.sha256(args.manifest_output.read_bytes()).hexdigest(),
    }
    args.manifest_output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote verified manifest with {len(manifest)} unique records to {args.manifest_output}")
print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
