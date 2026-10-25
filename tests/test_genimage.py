import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imad.data.genimage import create_genimage_manifest, discover_genimage, inspect_genimage


def make_image(path: Path, seed: int):
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (24, 24))
    image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(24 * 24)])
    image.save(path)


class GenImageTests(unittest.TestCase):
    def test_label_source_and_official_split_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_image(root / "ADM" / "train" / "ai" / "a.png", 1)
            make_image(root / "ADM" / "val" / "nature" / "b.png", 2)
            make_image(root / "ADM" / "test" / "nature" / "c.png", 3)
            rows = discover_genimage(root)
            by_id = {row["sample_id"]: row for row in rows}
            fake = by_id["ADM/train/ai/a.png"]
            real = by_id["ADM/val/nature/b.png"]
            self.assertEqual((fake["label"], fake["source_label"], fake["generator"], fake["source_split"]),
                             ("fake", "ai", "ADM", "train"))
            self.assertEqual((real["label"], real["source_split"]), ("real", "validation"))
            test_row = by_id["ADM/test/nature/c.png"]
            self.assertEqual(test_row["source_split"], "internal_test")

    def test_single_generator_root_uses_root_folder_name(self):
        with tempfile.TemporaryDirectory(prefix="ADM_") as tmp:
            root = Path(tmp)
            make_image(root / "train" / "ai" / "one.png", 1)
            row = discover_genimage(root)[0]
            self.assertEqual(row["generator"], root.name)

    def test_metadata_and_stratified_sample_without_decoding_every_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                for cls in ("ai", "nature"):
                    for index in range(3):
                        make_image(root / "ADM" / split / cls / f"{index}.png", index + (10 if cls == "ai" else 20))
            report = inspect_genimage(root, metadata_only=True)
            self.assertEqual(report["total_discovered"], 12)
            self.assertEqual(report["records"], [])
            sampled = inspect_genimage(root, sample_size=4, seed=5)
            self.assertEqual(sampled["audited"], 4)
            self.assertEqual({(r["source_split"], r["source_label"]) for r in sampled["records"]},
                             {("train", "ai"), ("train", "nature"), ("validation", "ai"), ("validation", "nature")})

    def test_manifest_keeps_duplicates_together_and_official_validation_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label_index, cls in enumerate(("ai", "nature")):
                for index in range(6):
                    make_image(root / "ADM" / "train" / cls / f"{index}.png", 100 * label_index + index)
                make_image(root / "ADM" / "val" / cls / "val.png", 200 + label_index)
            # Copy an exact training duplicate under a second generator directory.
            duplicate = root / "GLIDE" / "train" / "ai" / "copy.png"
            duplicate.parent.mkdir(parents=True)
            shutil.copyfile(root / "ADM" / "train" / "ai" / "0.png", duplicate)
            report = inspect_genimage(root)
            manifest = create_genimage_manifest(report["records"], seed=7)
            self.assertEqual({row["split"] for row in manifest}, {"train", "validation", "internal_test"})
            self.assertEqual(len({row["sample_id"] for row in manifest}), len(manifest))
            self.assertFalse(any(row["source_split"] == "validation" and row["split"] != "validation"
                                 for row in manifest))
            seen = {}
            for row in manifest:
                for fingerprint in (row["sha256"], "dhash:" + row["dhash"]):
                    seen.setdefault(fingerprint, set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in seen.values()))
            self.assertEqual(len(manifest), 14)  # one of 13? Exact duplicate removed from 14 discovered files.

    def test_multiple_generators_are_held_out_as_a_source_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generator_index, generator in enumerate(("ADM", "GLIDE")):
                for split in ("train", "val"):
                    for label_index, cls in enumerate(("ai", "nature")):
                        make_image(root / generator / split / cls / "sample.png",
                                   500 + generator_index * 100 + (split == "val") * 20 + label_index)
            report = inspect_genimage(root)
            manifest = create_genimage_manifest(report["records"], seed=41)
            test_sources = {r["generator"] for r in manifest if r["split"] == "internal_test"}
            train_sources = {r["generator"] for r in manifest if r["split"] == "train"}
            val_sources = {r["generator"] for r in manifest if r["split"] == "validation"}
            self.assertEqual(len(test_sources), 1)
            self.assertFalse(test_sources & train_sources)
            self.assertFalse(test_sources & val_sources)

    def test_duplicate_crossing_official_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repeated = root / "ADM" / "train" / "ai" / "same.png"
            make_image(repeated, 99)
            copy = root / "ADM" / "val" / "ai" / "same.png"
            copy.parent.mkdir(parents=True)
            shutil.copyfile(repeated, copy)
            # Additional unique samples provide enough training groups per class.
            make_image(root / "ADM" / "train" / "ai" / "other.png", 100)
            make_image(root / "ADM" / "train" / "nature" / "real.png", 101)
            make_image(root / "ADM" / "train" / "nature" / "real2.png", 102)
            make_image(root / "ADM" / "val" / "nature" / "realv.png", 103)
            report = inspect_genimage(root)
            with self.assertRaisesRegex(ValueError, "crosses official train/val"):
                create_genimage_manifest(report["records"])

    def test_unrecognized_layout_and_sample_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_image(root / "fake" / "x.png", 1)
            with self.assertRaisesRegex(ValueError, "Unrecognized GenImage"):
                discover_genimage(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                for cls in ("ai", "nature"):
                    make_image(root / "ADM" / split / cls / "x.png", hash(split + cls) & 0xffff)
            with self.assertRaises(ValueError):
                inspect_genimage(root, sample_size=0)

    def test_synthetic_preparation_cli_metadata_sample_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ADM"
            for split in ("train", "val"):
                for class_index, cls in enumerate(("ai", "nature")):
                    amount = 4 if split == "train" else 2
                    for index in range(amount):
                        make_image(root / split / cls / f"{index}.png",
                                   900 + class_index * 100 + index + (split == "val") * 10)
            repo = Path(__file__).resolve().parents[1]
            python = sys.executable
            metadata = subprocess.run([python, str(repo / "scripts" / "prepare_genimage.py"), str(root),
                                       "--metadata-only"], check=True, capture_output=True, text=True)
            self.assertIn('"metadata_only": true', metadata.stdout)
            sample = subprocess.run([python, str(repo / "scripts" / "prepare_genimage.py"), str(root),
                                     "--sample-size", "4"], check=True, capture_output=True, text=True)
            self.assertIn('"audited": 4', sample.stdout)
            manifest_path = Path(tmp) / "manifest.csv"
            subprocess.run([python, str(repo / "scripts" / "prepare_genimage.py"), str(root),
                            "--manifest-output", str(manifest_path)], check=True, capture_output=True, text=True)
            import csv
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in rows}, {"train", "validation", "internal_test"})
            self.assertEqual({row["label"] for row in rows}, {"fake", "real"})
            import json
            metadata = json.loads(manifest_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["split_counts"]["internal_test"], 2)


if __name__ == "__main__":
    unittest.main()
