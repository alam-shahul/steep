"""Cell type classification evaluator using frozen embeddings + logistic
regression."""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

CLASSIFICATION_METRIC_NAMES = ("accuracy", "macro_f1", "weighted_f1")


@dataclass
class ClassificationMetricAccumulator:
    scores: list[float] = field(default_factory=list)
    weighted_scores: list[tuple[float, int]] = field(default_factory=list)

    def add(self, score: float, count: int) -> None:
        self.scores.append(float(score))
        self.weighted_scores.append((float(score), int(count)))

    def mean(self) -> float | None:
        if not self.scores:
            return None
        return float(np.mean(self.scores))

    def weighted_mean(self) -> float | None:
        if not self.weighted_scores:
            return None
        total_weight = sum(weight for _, weight in self.weighted_scores)
        if total_weight == 0:
            return None
        return sum(score * weight for score, weight in self.weighted_scores) / total_weight


@dataclass
class ClassificationLabelStore:
    accuracy: ClassificationMetricAccumulator = field(default_factory=ClassificationMetricAccumulator)
    macro_f1: ClassificationMetricAccumulator = field(default_factory=ClassificationMetricAccumulator)
    weighted_f1: ClassificationMetricAccumulator = field(default_factory=ClassificationMetricAccumulator)

    def add_result(self, result: dict[str, float | int]) -> None:
        count = int(result["count"])
        self.accuracy.add(float(result["accuracy"]), count)
        self.macro_f1.add(float(result["macro_f1"]), count)
        self.weighted_f1.add(float(result["weighted_f1"]), count)

    def to_summary(self) -> dict[str, float | int | None]:
        if not self.accuracy.scores:
            return {
                "accuracy_mean": None,
                "accuracy_weighted_mean": None,
                "macro_f1_mean": None,
                "macro_f1_weighted_mean": None,
                "weighted_f1_mean": None,
                "weighted_f1_weighted_mean": None,
                "evaluated_slides": 0,
            }
        return {
            "accuracy_mean": self.accuracy.mean(),
            "accuracy_weighted_mean": self.accuracy.weighted_mean(),
            "macro_f1_mean": self.macro_f1.mean(),
            "macro_f1_weighted_mean": self.macro_f1.weighted_mean(),
            "weighted_f1_mean": self.weighted_f1.mean(),
            "weighted_f1_weighted_mean": self.weighted_f1.weighted_mean(),
            "evaluated_slides": len(self.accuracy.scores),
        }


def evaluate_slide_classification(
    labels_by_key: dict[str, np.ndarray],
    embeddings: np.ndarray,
    train_mask: np.ndarray,
    random_seed: int = 0,
    max_iter: int = 1000,
) -> dict[str, dict[str, float | int] | None]:
    """Train logistic regression per label_key on train cells, evaluate on test
    cells.

    Args:
        labels_by_key: dict of label_key -> array of labels for all cells in this slide.
        embeddings: (n_cells, embed_dim) embedding matrix for all cells in this slide.
        train_mask: boolean array of length n_cells; True = train, False = test.
        random_seed: seed for LogisticRegression.
        max_iter: max_iter for LogisticRegression.

    Returns:
        dict of label_key -> {accuracy, macro_f1, weighted_f1, count} or None if not evaluable.

    """
    warnings.simplefilter("ignore")

    test_mask = ~train_mask
    if train_mask.sum() < 2 or test_mask.sum() < 1:
        return {label_key: None for label_key in labels_by_key}

    X_train = embeddings[train_mask]
    X_test = embeddings[test_mask]

    results = {}
    for label_key, labels in labels_by_key.items():
        valid_train = train_mask & ~pd.isna(labels)
        valid_test = test_mask & ~pd.isna(labels)
        if valid_train.sum() < 2 or valid_test.sum() < 1:
            results[label_key] = None
            continue

        y_train = labels[valid_train]
        y_test = labels[valid_test]

        train_classes = np.unique(y_train)
        if len(train_classes) < 2:
            results[label_key] = None
            continue

        test_known_class_mask = np.isin(y_test, train_classes)
        if test_known_class_mask.sum() < 1:
            results[label_key] = None
            continue

        X_train_k = embeddings[valid_train]
        X_test_k = embeddings[valid_test][test_known_class_mask]
        y_test_k = y_test[test_known_class_mask]

        clf = LogisticRegression(
            max_iter=max_iter,
            n_jobs=-1,
            random_state=random_seed,
        )
        clf.fit(X_train_k, y_train)
        y_pred = clf.predict(X_test_k)

        results[label_key] = {
            "accuracy": float(accuracy_score(y_test_k, y_pred)),
            "macro_f1": float(f1_score(y_test_k, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test_k, y_pred, average="weighted", zero_division=0)),
            "count": int(len(y_test_k)),
            "n_test_dropped_unknown_class": int((~test_known_class_mask).sum()),
        }

    return results


def accumulate_classification_results(
    metric_store: dict[str, ClassificationLabelStore],
    slide_results: dict[str, dict[str, float | int] | None],
) -> None:
    for label_key, result in slide_results.items():
        if result is None:
            continue
        metric_store[label_key].add_result(result)


def summarize_classification_scores(
    label_keys: list[str] | tuple[str, ...],
    metric_store: dict[str, ClassificationLabelStore],
) -> dict[str, object]:
    metrics_by_label_key = {label_key: metric_store[label_key].to_summary() for label_key in label_keys}
    return {
        "label_keys": list(label_keys),
        "classification_metrics_by_label_key": metrics_by_label_key,
    }
