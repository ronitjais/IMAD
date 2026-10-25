"""GenImage-specific discovery, auditing, and leakage-aware manifests.

Expected layout is `<generator>/{train,val}/{ai,nature}/<image>`, matching the
official repository's documented directory tree. This module never downloads data.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .images import IMAGE_EXTENSIONS

CLASS_MAP = {"ai": "fake", "nature": "real"}
SPLIT_MAP = {"train": "train", "val": "validation", "test": "internal_test"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_genimage(root: str | Path) -> list[dict[str, str]]:
    """Parse documented GenImage paths; reject files outside that schema."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"GenImage directory does not exist: {base}")
    rows = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        relative = path.relative_to(base)
        parts = relative.parts
        matches = [i for i in range(len(parts) - 2)
                   if parts[i].lower() in SPLIT_MAP and parts[i + 1].lower() in CLASS_MAP]
        if len(matches) != 1:
            raise ValueError(f"Unrecognized GenImage path schema: {relative.as_posix()}")
        split_index = matches[0]
        # A single-generator archive may be extracted directly as train/val.
        source = parts[split_index - 1] if split_index else base.name
        source_class = parts[split_index + 1].lower()
        rows.append({"sample_id": relative.as_posix(), "path": str(path),
                     "generator": source, "source_split": SPLIT_MAP[parts[split_index].lower()],
                     "source_label": source_class, "label": CLASS_MAP[source_class]})
    if not rows:
        raise ValueError("No GenImage files found; expected <generator>/{train,val}/{ai,nature}/<image>")
    return rows


def _dhash(path: Path) -> str:
    """64-bit difference hash: a practical near-duplicate indicator, not identity proof."""
    with Image.open(path) as image:
        pixels = list(ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS).tobytes())
    bits = [pixels[row * 9 + col] > pixels[row * 9 + col + 1]
            for row in range(8) for col in range(8)]
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def inspect_genimage(root: str | Path, *, sample_size: int | None = None,
                     seed: int = 42, metadata_only: bool = False) -> dict[str, Any]:
    """Count schema/class/source/split metadata; optionally verify a sample or all files."""
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be positive")
    records = discover_genimage(root)
    counts = Counter((r["generator"], r["source_split"], r["source_label"]) for r in records)
    selected = records
    sampled = sample_size is not None and 0 < sample_size < len(records)
    if sampled:
        # Reserve at least one sample for each official split/class stratum.
        strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in records:
            strata[(row["source_split"], row["source_label"])].append(row)
        if sample_size < len(strata):
            raise ValueError(f"sample_size must be at least {len(strata)} to cover every split/class")
        rng = random.Random(seed)
        buckets = list(strata.values())
        selected = []
        for bucket in buckets:
            rng.shuffle(bucket)
            selected.append(bucket.pop())
        remainder = [row for bucket in buckets for row in bucket]
        selected.extend(rng.sample(remainder, min(sample_size - len(selected), len(remainder))))
    corrupt = []
    audited = []
    dimensions: Counter[str] = Counter()
    if not metadata_only:
        for row in selected:
            path = Path(row["path"])
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                content_hash = _sha256_file(path)
                perceptual_hash = _dhash(path)
                dimensions[f"{width}x{height}"] += 1
                audited.append({**row, "width": width, "height": height,
                                "sha256": content_hash, "dhash": perceptual_hash})
            except (UnidentifiedImageError, OSError, ValueError):
                corrupt.append(row["sample_id"])
    return {
        "root": str(Path(root).expanduser().resolve()),
        "schema": "<generator>/{train,val[,test]}/{ai,nature}/<image>",
        "label_mapping": CLASS_MAP.copy(),
        "source_split_mapping": SPLIT_MAP.copy(),
        "total_discovered": len(records), "audited": len(audited),
        "sampled": sampled, "metadata_only": metadata_only,
        "counts_by_generator_split_source_label": [
            {"generator": generator, "source_split": split, "source_label": label,
             "imad_label": CLASS_MAP[label], "count": count}
            for (generator, split, label), count in sorted(counts.items())],
        "dimension_counts": dict(sorted(dimensions.items())),
        "corrupt_count": len(corrupt), "corrupt_files": corrupt,
        "records": audited,
        "note": "Mapped labels: ai=fake, nature=real (ImageNet-collected); this is a dataset task label, not proof of authenticity.",
    }


def create_genimage_manifest(records: list[dict[str, Any]], *, seed: int = 42,
                              internal_test_fraction: float = 0.15) -> list[dict[str, str]]:
    """Preserve official val; carve internal test from official train by hash group.

    Exact SHA-256 and equal 64-bit dHash records are kept in one group. dHash
    collisions are conservative duplicate indicators and can reduce effective data.
    Cross-official-split duplicate groups are rejected rather than silently moved.
    """
    if not 0 < internal_test_fraction < 1:
        raise ValueError("internal_test_fraction must be strictly between 0 and 1")
    if not records:
        raise ValueError("No audited records supplied")
    if any(not r.get("sha256") or not r.get("dhash") for r in records):
        raise ValueError("A full image audit with SHA-256 and dHash is required before splitting")
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_sha: dict[str, int] = {}
    seen_dhash: dict[tuple[str, str], int] = {}
    for i, row in enumerate(records):
        if row["sha256"] in seen_sha:
            union(i, seen_sha[row["sha256"]])
        else:
            seen_sha[row["sha256"]] = i
        # Equal dHash is a conservative cue only within a class; across classes
        # it can be a natural collision (for example, flat images).
        key = (row["label"], row["dhash"])
        if key in seen_dhash:
            union(i, seen_dhash[key])
        else:
            seen_dhash[key] = i
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row)
    by_label_split: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for group in groups.values():
        labels = {row["label"] for row in group}
        for row in group:
            if row.get("source_label") not in CLASS_MAP or row["label"] != CLASS_MAP[row["source_label"]]:
                raise ValueError("Source class and normalized label mapping disagree")
        splits = {row["source_split"] for row in group}
        if len(labels) != 1:
            raise ValueError("An exact/perceptual duplicate group has conflicting class labels")
        if len(splits) != 1:
            raise ValueError("A duplicate group crosses official train/val; remove leakage before splitting")
        split = next(iter(splits))
        if split not in {"train", "validation", "internal_test"}:
            raise ValueError(f"Unsupported official source split: {split}")
        by_label_split[(next(iter(labels)), split)].append(group)
    rng = random.Random(seed)
    output = []
    has_official_test = any(source_split == "internal_test" for _, source_split in by_label_split)
    if not has_official_test:
        generators_by_label: dict[str, set[str]] = defaultdict(set)
        for values in by_label_split.values():
            for group in values:
                generators_by_label[group[0]["label"]].add(group[0]["generator"])
        all_labels = set(generators_by_label)
        candidates = []
        for heldout in sorted(set.union(*generators_by_label.values()) if generators_by_label else set()):
            heldout_labels = {label for label, split in by_label_split
                              if any(group[0]["generator"] == heldout
                                     for group in by_label_split[(label, split)])}
            remaining = set.union(*(sources - {heldout} for sources in generators_by_label.values()))
            train_labels = {label for (label, split), values in by_label_split.items()
                            if split == "train" and any(group[0]["generator"] != heldout for group in values)}
            val_labels = {label for (label, split), values in by_label_split.items()
                          if split == "validation" and any(group[0]["generator"] != heldout for group in values)}
            if heldout_labels == all_labels and remaining and train_labels == all_labels and val_labels == all_labels:
                candidates.append(heldout)
        if candidates:
            heldout = rng.choice(candidates)
            for (_, source_split), groups_for_class in by_label_split.items():
                for group in groups_for_class:
                    split = ("internal_test" if group[0]["generator"] == heldout else
                             "validation" if source_split == "validation" else "train")
                    row = min(group, key=lambda item: item["sample_id"])
                    output.append({"sample_id": row["sample_id"], "path": row["path"],
                                   "label": row["label"], "source_label": row["source_label"],
                                   "generator": row["generator"], "source_split": source_split,
                                   "split": split, "sha256": row["sha256"], "dhash": row["dhash"]})
            _validate_manifest(output)
            return sorted(output, key=lambda row: row["sample_id"])
    for (label, source_split), groups_for_class in sorted(by_label_split.items()):
        rng.shuffle(groups_for_class)
        n = len(groups_for_class)
        if source_split in {"validation", "internal_test"}:
            assignments = [source_split] * n
        elif has_official_test:
            assignments = ["train"] * n
        else:
            if n < 2:
                raise ValueError(f"At least two unique hash groups required in train class {label}")
            n_test = min(n - 1, max(1, round(n * internal_test_fraction)))
            assignments = ["internal_test" if i < n_test else "train" for i in range(n)]
        for group, split in zip(groups_for_class, assignments):
            # One representative per perceptual/SHA duplicate group avoids duplicate weighting.
            row = min(group, key=lambda item: item["sample_id"])
            output.append({"sample_id": row["sample_id"], "path": row["path"], "label": label,
                           "source_label": row["source_label"], "generator": row["generator"],
                           "source_split": source_split, "split": split,
                           "sha256": row["sha256"], "dhash": row["dhash"]})
    _validate_manifest(output)
    return sorted(output, key=lambda row: row["sample_id"])


def _validate_manifest(rows: list[dict[str, str]]) -> None:
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest sample identifiers must be unique")
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hash_splits[row["sha256"]].add(row["split"])
        hash_splits["dhash:" + row["label"] + ":" + row["dhash"]].add(row["split"])
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise ValueError("Exact or equal-dHash duplicates cross manifest splits")
    required = {"train", "validation", "internal_test"}
    found = {row["split"] for row in rows}
    if not required.issubset(found):
        raise ValueError(f"Manifest must contain {sorted(required)}; found {sorted(found)}")
