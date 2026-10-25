"""Train a ResNet-18 transfer-learning baseline from a split CSV.

The pretrained ImageNet weights are fetched by torchvision on first run. Use only
after confirming dataset label semantics and reviewing the split manifest.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


class ManifestDataset(Dataset):
    def __init__(self, rows, label_to_id, image_size=224):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/image_baseline.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Use at most 32 images per split")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if any(row.get("source_split_hint") or row.get("source_label") for row in rows):
        raise ValueError("This generic trainer does not preserve GenImage holdouts; use train_genimage_baseline.py")
    paths_by_split = {}
    for row in rows:
        paths_by_split.setdefault(row["split"], set()).add(str(Path(row["path"]).resolve()))
    splits = list(paths_by_split)
    for index, split in enumerate(splits):
        for other in splits[index + 1:]:
            overlap = paths_by_split[split] & paths_by_split[other]
            if overlap:
                raise ValueError(f"Image paths overlap between {split} and {other}")
    labels = sorted({r["label"] for r in rows})
    if len(labels) < 2:
        raise ValueError("Training requires at least two label classes")
    label_to_id = {label: i for i, label in enumerate(labels)}
    datasets = {}
    for split in ("train", "validation"):
        split_rows = [r for r in rows if r["split"] == split]
        if args.smoke_test:
            split_rows = split_rows[:32]
        if not split_rows:
            raise ValueError(f"No records in {split} split")
        datasets[split] = DataLoader(ManifestDataset(split_rows, label_to_id, args.image_size),
            batch_size=args.batch_size, shuffle=(split == "train"), num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    history = []
    for epoch in range(args.epochs):
        epoch_record = {"epoch": epoch + 1}
        for phase in ("train", "validation"):
            model.train(phase == "train")
            loss_sum = count = correct = 0
            for images, targets in datasets[phase]:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    logits = model(images)
                    loss = criterion(logits, targets)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                loss_sum += loss.item() * targets.size(0)
                count += targets.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
            epoch_record[f"{phase}_loss"] = loss_sum / count
            epoch_record[f"{phase}_accuracy"] = correct / count
        history.append(epoch_record)
        print(json.dumps(epoch_record))
        if epoch_record["validation_loss"] < best_loss:
            best_loss = epoch_record["validation_loss"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.cpu().state_dict(), "labels": labels,
                "architecture": "resnet18", "image_size": args.image_size,
                "seed": args.seed, "history": history}, args.output)
            model.to(device)
    args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved validation-selected checkpoint: {args.output} ({device})")


if __name__ == "__main__":
    main()
