"""Dependency-light classification metrics with explicit positive-label semantics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], *,
                           positive_label: str, positive_scores: Sequence[float] | None = None
                           ) -> dict[str, Any]:
    """Compute binary/multiclass accuracy, per-class metrics, and optional binary AUC.

    `positive_scores` must be scores for `positive_label` (not a probability unless
    independently calibrated). ROC-AUC is omitted for single-class truth labels.
    """
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    if positive_scores is not None and len(positive_scores) != len(y_true):
        raise ValueError("positive_scores length must match y_true")
    labels = sorted(set(y_true) | set(y_pred))
    support = Counter(y_true)
    per_class = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0, "support": support[label]}
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    tp = sum(t == positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fp = sum(t != positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_label and p != positive_label for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    matrix = {actual: {predicted: sum(t == actual and p == predicted
                for t, p in zip(y_true, y_pred)) for predicted in labels} for actual in labels}
    result = {"positive_label": positive_label, "accuracy": correct / len(y_true),
              "precision": precision, "recall": recall,
              "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
              "labels": labels, "confusion_matrix": matrix, "per_class": per_class,
              "score_semantics": "uncalibrated score" if positive_scores is not None else None}
    if positive_scores is not None and len(set(y_true)) == 2:
        result["roc_auc"] = _binary_auc(y_true, positive_scores, positive_label)
    else:
        result["roc_auc"] = None
    return result


def _binary_auc(y_true: Sequence[str], scores: Sequence[float], positive_label: str) -> float | None:
    entries = [(float(score), label == positive_label) for label, score in zip(y_true, scores)]
    positive_count = sum(is_positive for _, is_positive in entries)
    negative_count = len(entries) - positive_count
    if not positive_count or not negative_count:
        return None
    if any(not (float("-inf") < score < float("inf")) for score, _ in entries):
        raise ValueError("Scores must be finite")
    entries.sort(key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(entries):
        end = index + 1
        while end < len(entries) and entries[end][0] == entries[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2  # average of 1-based ranks [index+1, end]
        positive_rank_sum += average_rank * sum(is_positive for _, is_positive in entries[index:end])
        index = end
    return ((positive_rank_sum - positive_count * (positive_count + 1) / 2)
            / (positive_count * negative_count))
