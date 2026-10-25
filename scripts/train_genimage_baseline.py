"""Train, validation-select, and internally test ResNet-18 from a GenImage manifest."""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imad.evaluation.metrics import classification_metrics


class ManifestDataset(Dataset):
    def __init__(self, rows, label_to_id, image_size):
        self.rows = rows
        self.label_to_id = label_to_id
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_id[row["label"]]


def _predict(model, loader, device, labels, positive_label):
    model.eval()
    true_ids, pred_ids, positive_scores = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
            true_ids.extend(targets.tolist())
            pred_ids.extend(probabilities.argmax(1).tolist())
            positive_scores.extend(probabilities[:, labels.index(positive_label)].tolist())
    y_true = [labels[i] for i in true_ids]
    y_pred = [labels[i] for i in pred_ids]
    metrics = classification_metrics(y_true, y_pred, positive_label=positive_label,
                                    positive_scores=positive_scores)
    return metrics, y_true, y_pred, positive_scores


def _plot_training_curves(history, output_dir):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [entry["epoch"] for entry in history]
    for key in ("train_loss", "validation_loss"):
        axes[0].plot(epochs, [entry[key] for entry in history], label=key)
    for key in ("train_accuracy", "validation_accuracy"):
        axes[1].plot(epochs, [entry[key] for entry in history], label=key)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/genimage_resnet18"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true", help="Use at most 16 images per split")
    parser.add_argument("--resume", action="store_true", help="Resume trusted checkpoint in output directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = sorted({row["label"] for row in rows})
    if labels != ["fake", "real"]:
        raise ValueError(f"Expected GenImage classes ['fake', 'real']; found {labels}")
    label_to_id = {label: index for index, label in enumerate(labels)}
    split_rows = {split: [r for r in rows if r["split"] == split]
                  for split in ("train", "validation", "internal_test")}
    seen_paths, seen_hashes = {}, {}
    for row in rows:
        path = str(Path(row["path"]).resolve())
        if path in seen_paths and seen_paths[path] != row["split"]:
            raise ValueError("Dataset leakage: an image path appears in multiple splits")
        seen_paths[path] = row["split"]
        for fingerprint in (row.get("sha256", ""),
                            "dhash:" + row.get("label", "") + ":" + row.get("dhash", "")):
            if not fingerprint or fingerprint.endswith(":"):
                continue
            if fingerprint in seen_hashes and seen_hashes[fingerprint] != row["split"]:
                raise ValueError("Dataset leakage: a duplicate fingerprint crosses splits")
            seen_hashes[fingerprint] = row["split"]
    for split, values in split_rows.items():
        if args.smoke_test:
            values = [row for label in labels for row in values if row["label"] == label][:16]
            split_rows[split] = values
        if not values:
            raise ValueError(f"No records in required {split} split")
        present = {row["label"] for row in values}
        if present != set(labels):
            raise ValueError(f"{split} split must contain both labels; found {sorted(present)}")
    for split, values in split_rows.items():
        print(f"{split} class counts: {dict(Counter(r['label'] for r in values))}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(ManifestDataset(values, label_to_id, args.image_size),
                          batch_size=args.batch_size, shuffle=(split == "train"),
                          num_workers=args.workers, pin_memory=(device.type == "cuda"),
                          generator=generator if split == "train" else None)
        for split, values in split_rows.items()
    }
    pretrained_weights = None if args.resume else models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=pretrained_weights)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.to(device)
    counts = Counter(row["label"] for row in split_rows["train"])
    weights = torch.tensor([len(split_rows["train"]) / (len(labels) * counts[label])
                            for label in labels], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    completion_path = args.output_dir / "experiment_complete.json"
    if completion_path.exists():
        raise FileExistsError("This output directory already has an evaluated internal test; choose a new output directory")
    history, best_loss, start_epoch = [], float("inf"), 0
    resume_config = None
    if args.resume:
        latest_path = args.output_dir / "last_checkpoint.pt"
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot resume; latest checkpoint is missing: {latest_path}")
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved.get("labels") != labels:
            raise ValueError("Checkpoint label order does not match this dataset")
        if bool(saved.get("config", {}).get("smoke_test")) != args.smoke_test:
            raise ValueError("Cannot resume across smoke-test and full-run configurations")
        resume_config = saved.get("config", {})
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = saved.get("history", [])
        best_loss = saved["best_validation_loss"]
        start_epoch = saved["epoch"]
        model.to(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    config = {"architecture": "resnet18", "labels": labels, "label_to_id": label_to_id,
              "pretrained_weights": pretrained_weights.name if pretrained_weights else "loaded_from_resume_checkpoint",
              "positive_label": "fake", "image_size": args.image_size,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "target_epochs": args.epochs, "seed": args.seed, "device": str(device),
              "manifest": str(args.manifest.resolve()), "smoke_test": args.smoke_test,
              "manifest_sha256": manifest_sha256,
              "torch_version": torch.__version__, "torchvision_version": __import__("torchvision").__version__}
    if resume_config:
        invariant_keys = ("image_size", "batch_size", "learning_rate", "seed", "manifest_sha256", "smoke_test")
        changed = [key for key in invariant_keys if resume_config.get(key) != config.get(key)]
        if changed:
            raise ValueError(f"Resume configuration differs for: {', '.join(changed)}")
    if args.resume and start_epoch >= args.epochs:
        raise ValueError("Checkpoint already reached the target epoch; choose a new target or output directory")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "label_mapping.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        entry = {"epoch": epoch + 1}
        for phase in ("train", "validation"):
            model.train(phase == "train")
            loss_sum = correct = count = 0
            for images, targets in loaders[phase]:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    logits = model(images)
                    loss = criterion(logits, targets)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                loss_sum += loss.item() * len(targets)
                count += len(targets)
                correct += (logits.argmax(1) == targets).sum().item()
            entry[f"{phase}_loss"] = loss_sum / count
            entry[f"{phase}_accuracy"] = correct / count
        validation, _, _, _ = _predict(model, loaders["validation"], device, labels, "fake")
        validation["evaluation_split"] = "validation"
        entry["validation_metrics"] = validation
        history.append(entry)
        (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(entry))
        if entry["validation_loss"] < best_loss:
            best_loss = entry["validation_loss"]
            torch.save({"state_dict": model.cpu().state_dict(), "labels": labels,
                        "epoch": epoch + 1, "best_validation_loss": best_loss,
                        "history": history, "config": config}, checkpoint_path)
            model.to(device)
        latest = {"state_dict": model.cpu().state_dict(), "optimizer_state": optimizer.state_dict(),
                  "labels": labels, "epoch": epoch + 1, "best_validation_loss": best_loss,
                  "history": history, "config": config}
        torch.save(latest, args.output_dir / "last_checkpoint.pt")
        model.to(device)
    if args.smoke_test:
        _plot_training_curves(history, args.output_dir)
        print("Smoke fit completed. Internal test was not opened or evaluated.")
        print(f"Smoke artifacts saved under {args.output_dir.resolve()}")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    internal_test, y_true, y_pred, scores = _predict(model, loaders["internal_test"], device,
                                                     labels, "fake")
    internal_test["evaluation_split"] = "internal_test"
    test_generators = [row["generator"] for row in split_rows["internal_test"]]
    internal_test["by_generator"] = {}
    for generator_name in sorted(set(test_generators)):
        selected = [i for i, name in enumerate(test_generators) if name == generator_name]
        internal_test["by_generator"][generator_name] = classification_metrics(
            [y_true[i] for i in selected], [y_pred[i] for i in selected], positive_label="fake",
            positive_scores=[scores[i] for i in selected])
    (args.output_dir / "internal_test_metrics.json").write_text(
        json.dumps(internal_test, indent=2), encoding="utf-8")
    completion_path.write_text(json.dumps({"evaluation_split": "internal_test",
        "checkpoint_epoch": checkpoint["epoch"], "manifest_sha256": manifest_sha256}, indent=2), encoding="utf-8")
    with (args.output_dir / "internal_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "generator", "y_true", "y_pred", "positive_score"])
        writer.writerows((row["sample_id"], row["generator"], truth, pred, score)
                         for row, truth, pred, score in zip(split_rows["internal_test"], y_true, y_pred, scores))
    import matplotlib.pyplot as plt
    matrix = internal_test["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow([[matrix[a][p] for p in labels] for a in labels], cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted", ylabel="Actual", title="Internal test confusion matrix")
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            ax.text(j, i, matrix[actual][predicted], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.output_dir / "internal_test_confusion_matrix.png", dpi=160)
    plt.close(fig)
    _plot_training_curves(history, args.output_dir)
    print(json.dumps(internal_test, indent=2))
    print(f"Artifacts saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
