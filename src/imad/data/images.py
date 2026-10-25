"""Image dataset discovery, inspection, and reproducible split manifests."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "dev", "test"}


def discover_images(root: str | Path) -> list[dict[str, Any]]:
    """Discover images and use immediate parent names as provisional labels.

    The immediate parent name is a provisional class candidate. Earlier path
    components are preserved as source and split hints for manual review.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Image dataset directory does not exist: {base}")
    rows = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        relative = path.relative_to(base)
        components = relative.parts[:-1]
        label = components[-1] if components else "unlabeled"
        split = next((part for part in components if part.lower() in SPLIT_NAMES), None)
        source = components[0] if len(components) > 1 else None
        rows.append({"path": str(path), "relative_path": relative.as_posix(), "label": label,
                     "source_hint": source, "source_split_hint": split})
    return rows


def inspect_images(root: str | Path, sample_size: int | None = None, seed: int = 42) -> dict[str, Any]:
    """Inspect dimensions, decodability, label counts, exact duplicates and sample."""
    rows = discover_images(root)
    total_found = len(rows)
    if sample_size is not None and sample_size > 0 and sample_size < len(rows):
        rows = random.Random(seed).sample(rows, sample_size)
    dimensions: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    digest_paths: dict[str, list[str]] = defaultdict(list)
    corrupt: list[str] = []
    inspected = []
    for row in rows:
        labels[row["label"]] += 1
        path = Path(row["path"])
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            dimensions[f"{width}x{height}"] += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digest_paths[digest].append(row["relative_path"])
            inspected.append({**row, "width": width, "height": height, "sha256": digest})
        except (UnidentifiedImageError, OSError, ValueError):
            corrupt.append(row["relative_path"])
    duplicates = [paths for paths in digest_paths.values() if len(paths) > 1]
    return {
        "root": str(Path(root).expanduser().resolve()),
        "total_discovered": total_found,
        "inspected": len(rows),
        "sampled": len(rows) < total_found,
        "labels": dict(sorted(labels.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "corrupt_count": len(corrupt),
        "corrupt_files": corrupt,
        "exact_duplicate_groups": duplicates,
        "records": inspected,
        "note": "Directory-derived labels are provisional. Near-duplicate detection is not performed.",
    }


def create_split_manifest(records: list[dict[str, Any]], *, seed: int = 42,
                          validation_fraction: float = 0.15, test_fraction: float = 0.15) -> list[dict[str, str]]:
    """Create seeded, label-stratified splits after excluding corrupt/duplicate records.

    Exact duplicate groups are assigned as a unit using sha256. Near-duplicate or
    source-aware splitting requires suitable metadata and is not inferred here.
    """
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be non-negative and sum to less than 1")
    usable = [r for r in records if r.get("sha256") and r.get("label")]
    official_names = {"train": "train", "training": "train", "val": "validation",
                      "valid": "validation", "validation": "validation", "dev": "validation",
                      "test": "test"}
    if usable and all((r.get("source_split_hint") or "").lower() in official_names for r in usable):
        digest_splits: dict[str, set[str]] = defaultdict(set)
        for row in usable:
            digest_splits[row["sha256"]].add(official_names[(row.get("source_split_hint") or "").lower()])
        if any(len(values) > 1 for values in digest_splits.values()):
            raise ValueError("An exact duplicate appears in multiple source splits; resolve leakage first")
        return [{"path": r["path"], "label": r["label"],
                 "split": official_names[(r.get("source_split_hint") or "").lower()],
                 "source_hint": r.get("source_hint", ""),
                 "source_split_hint": r.get("source_split_hint", "")} for r in usable]
    digest_labels: dict[str, set[str]] = defaultdict(set)
    for row in usable:
        digest_labels[row["sha256"]].add(row["label"])
    conflicts = [digest for digest, values in digest_labels.items() if len(values) > 1]
    if conflicts:
        raise ValueError(f"Exact duplicate images have conflicting labels ({len(conflicts)} digest groups)")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        grouped[row.get("sha256", row["path"])].append(row)
    by_label: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in grouped.values():
        by_label[group[0]["label"]].append(group)
    rng = random.Random(seed)
    output = []
    for label, groups in sorted(by_label.items()):
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(n * test_fraction)
        n_val = int(n * validation_fraction)
        if n >= 3:
            n_test = max(1, n_test) if test_fraction else 0
            n_val = max(1, n_val) if validation_fraction else 0
            while n_test + n_val >= n:
                if n_val > 0: n_val -= 1
                elif n_test > 0: n_test -= 1
        for i, group in enumerate(groups):
            split = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
            output.extend({"path": row["path"], "label": label, "split": split,
                           "source_hint": row.get("source_hint", ""),
                           "source_split_hint": row.get("source_split_hint", "")} for row in group)
    return output
