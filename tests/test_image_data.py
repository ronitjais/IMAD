from PIL import Image
import unittest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imad.data.images import create_split_manifest, discover_images, inspect_images
from imad.evaluation.metrics import classification_metrics


class ImageDataTests(unittest.TestCase):
    def test_discovery_and_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, color in (("real", "blue"), ("fake", "red")):
                (root / label).mkdir()
                Image.new("RGB", (12, 8), color=color).save(root / label / "one.png")
            rows = discover_images(root)
            report = inspect_images(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(report["labels"], {"fake": 1, "real": 1})
            self.assertEqual(report["dimensions"], {"12x8": 2})
            self.assertEqual(report["corrupt_count"], 0)

    def test_nested_source_split_and_class_hints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "GeneratorA" / "train" / "ai" / "example.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2), color="red").save(image_path)
            row = discover_images(root)[0]
            self.assertEqual(row["label"], "ai")
            self.assertEqual(row["source_hint"], "GeneratorA")
            self.assertEqual(row["source_split_hint"], "train")

    def test_official_source_split_is_preserved(self):
        rows = [
            {"path": "a.png", "label": "ai", "sha256": "a", "source_split_hint": "train"},
            {"path": "b.png", "label": "nature", "sha256": "b", "source_split_hint": "val"},
        ]
        manifest = create_split_manifest(rows)
        self.assertEqual([row["split"] for row in manifest], ["train", "validation"])


    def test_corrupt_images_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fake").mkdir()
            (root / "fake" / "broken.jpg").write_bytes(b"not an image")
            self.assertEqual(inspect_images(root)["corrupt_files"], ["fake/broken.jpg"])


    def test_split_is_reproducible_and_keeps_exact_duplicates_together(self):
        rows = [
            {"path": f"{i}.jpg", "label": "fake" if (i // 2) % 2 else "real", "sha256": str(i // 2)}
            for i in range(20)
        ]
        first = create_split_manifest(rows, seed=8)
        second = create_split_manifest(rows, seed=8)
        self.assertEqual(first, second)
        splits_by_digest = {}
        for row in first:
            digest = str(int(row["path"].split(".")[0]) // 2)
            splits_by_digest.setdefault(digest, set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_digest.values()))


    def test_invalid_split_fractions_rejected(self):
        with self.assertRaises(ValueError):
            create_split_manifest([], validation_fraction=0.7, test_fraction=0.4)

    def test_metrics_and_auc_positive_label_mapping(self):
        result = classification_metrics(["real", "fake", "fake", "real"],
            ["real", "fake", "real", "real"], positive_label="fake",
            positive_scores=[0.1, 0.9, 0.8, 0.2])
        self.assertEqual(result["confusion_matrix"]["fake"]["fake"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["roc_auc"], 1.0)

    def test_metrics_reject_mismatched_inputs_and_omit_single_class_auc(self):
        with self.assertRaises(ValueError):
            classification_metrics(["real"], [], positive_label="fake")
        result = classification_metrics(["real"], ["real"], positive_label="fake", positive_scores=[0.1])
        self.assertIsNone(result["roc_auc"])

    def test_auc_handles_tied_scores(self):
        result = classification_metrics(["fake", "real", "fake", "real"],
            ["fake", "real", "real", "fake"], positive_label="fake",
            positive_scores=[0.5, 0.5, 0.2, 0.2])
        self.assertEqual(result["roc_auc"], 0.5)

if __name__ == "__main__":
    unittest.main()
