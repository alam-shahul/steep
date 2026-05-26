"""Cell type classification evaluator using frozen embeddings + kNN."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

CLASSIFICATION_METRIC_NAMES = ("accuracy", "macro_f1", "weighted_f1")

# Number of neighbors used by the kNN classifier.
DEFAULT_N_NEIGHBORS = 5
MAX_CONFUSION_MATRIX_CLASSES = 100


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
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
) -> dict[str, dict[str, float | int] | None]:
    """Fit a kNN classifier per label_key on train cells, evaluate on test
    cells.

    Args:
        labels_by_key: dict of label_key -> array of labels for all cells in this slide.
        embeddings: (n_cells, embed_dim) embedding matrix for all cells in this slide.
        train_mask: boolean array of length n_cells; True = train, False = test.
        random_seed: unused for kNN.
        max_iter: unused for kNN.
        n_neighbors: k for kNN. Default 5.

    Returns:
        dict of label_key -> {accuracy, macro_f1, weighted_f1, count, ...} or None
        if not evaluable.

    """
    del random_seed, max_iter

    warnings.simplefilter("ignore")

    test_mask = ~train_mask
    if train_mask.sum() < 2 or test_mask.sum() < 1:
        return {label_key: None for label_key in labels_by_key}

    results: dict[str, dict[str, float | int] | None] = {}
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

        effective_k = min(int(n_neighbors), int(X_train_k.shape[0]))
        clf = KNeighborsClassifier(n_neighbors=effective_k, n_jobs=1)
        clf.fit(X_train_k, y_train)
        y_pred = clf.predict(X_test_k)

        results[label_key] = {
            "accuracy": float(accuracy_score(y_test_k, y_pred)),
            "macro_f1": float(f1_score(y_test_k, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test_k, y_pred, average="weighted", zero_division=0)),
            "count": int(len(y_test_k)),
            "n_classes_train": int(len(train_classes)),
            "n_train": int(X_train_k.shape[0]),
            "n_neighbors": int(effective_k),
            "classifier": "KNeighborsClassifier",
            "n_test_dropped_unknown_class": int((~test_known_class_mask).sum()),
        }

    return results


def _labels_for_records(records, label_key: str) -> np.ndarray:
    return np.concatenate([record.labels_by_key[label_key] for record in records if label_key in record.labels_by_key])


def _embeddings_for_records(records, label_key: str) -> np.ndarray:
    return np.concatenate([record.embeddings for record in records if label_key in record.labels_by_key], axis=0)


def _save_confusion_matrix(
    y_test,
    y_pred,
    classes,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    width = max(6.0, min(18.0, 0.4 * len(classes) + 4.0))
    fig, ax = plt.subplots(figsize=(width, width))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="OrRd", ax=ax, colorbar=True, xticks_rotation="vertical")
    ax.set_title("kNN cell type classification")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in str(value))


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    if hasattr(values, "get"):
        return values.get()
    return np.asarray(values)


def _predict_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int,
    backend: str,
    log_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if backend == "sklearn":
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1)
        if log_prefix is not None:
            logger.info("{} fitting sklearn kNN", log_prefix)
        clf.fit(X_train, y_train)
        if log_prefix is not None:
            logger.info("{} predicting with sklearn kNN", log_prefix)
        return clf.predict(X_test), clf.classes_, "KNeighborsClassifier"

    if backend not in {"rapids", "cuml"}:
        raise ValueError(f"Unsupported classification backend: {backend}")

    try:
        from cuml.neighbors import KNeighborsClassifier as CuMLKNeighborsClassifier
    except ImportError as exc:
        raise ImportError(
            "RAPIDS classification backend requires cuML. Install it with `uv sync --group rapids-cu12`.",
        ) from exc

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train).astype(np.int32)
    clf = CuMLKNeighborsClassifier(n_neighbors=n_neighbors, output_type="numpy")
    if log_prefix is not None:
        logger.info("{} fitting cuML kNN", log_prefix)
    clf.fit(X_train.astype(np.float32, copy=False), y_train_encoded)
    if log_prefix is not None:
        logger.info("{} predicting with cuML kNN", log_prefix)
    y_pred_encoded = _to_numpy(clf.predict(X_test.astype(np.float32, copy=False))).astype(np.int64, copy=False)
    if log_prefix is not None:
        logger.info("{} decoding cuML predictions", log_prefix)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    return y_pred, label_encoder.classes_, "CuMLKNeighborsClassifier"


def evaluate_split_classification(
    train_records,
    test_records,
    label_keys: list[str] | tuple[str, ...],
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    confusion_matrix_dir: str | Path | None = None,
    stage: str = "evaluation",
    backend: str = "sklearn",
) -> dict[str, dict[str, float | int | str] | None]:
    """Fit kNN on train-slide embeddings and evaluate on test-slide
    embeddings."""
    warnings.simplefilter("ignore")
    backend = str(backend).lower()

    results: dict[str, dict[str, float | int | str] | None] = {}
    for label_key in label_keys:
        logger.info("{} classification for {} using {}", stage.capitalize(), label_key, backend)
        if not any(label_key in record.labels_by_key for record in train_records):
            logger.info("{} classification for {} skipped: no train labels", stage.capitalize(), label_key)
            results[label_key] = None
            continue
        if not any(label_key in record.labels_by_key for record in test_records):
            logger.info("{} classification for {} skipped: no test labels", stage.capitalize(), label_key)
            results[label_key] = None
            continue

        X_train = _embeddings_for_records(train_records, label_key)
        X_test = _embeddings_for_records(test_records, label_key)
        y_train = _labels_for_records(train_records, label_key)
        y_test = _labels_for_records(test_records, label_key)

        valid_train = ~pd.isna(y_train)
        valid_test = ~pd.isna(y_test)
        if valid_train.sum() < 2 or valid_test.sum() < 1:
            logger.info(
                "{} classification for {} skipped: insufficient valid cells (train={}, test={})",
                stage.capitalize(),
                label_key,
                int(valid_train.sum()),
                int(valid_test.sum()),
            )
            results[label_key] = None
            continue

        X_train = X_train[valid_train]
        X_test = X_test[valid_test]
        y_train = y_train[valid_train]
        y_test = y_test[valid_test]

        train_classes = np.unique(y_train)
        if len(train_classes) < 2:
            logger.info(
                "{} classification for {} skipped: fewer than two train classes",
                stage.capitalize(),
                label_key,
            )
            results[label_key] = None
            continue

        test_known_class_mask = np.isin(y_test, train_classes)
        if test_known_class_mask.sum() < 1:
            logger.info(
                "{} classification for {} skipped: no test labels seen during training",
                stage.capitalize(),
                label_key,
            )
            results[label_key] = None
            continue

        X_test_known = X_test[test_known_class_mask]
        y_test_known = y_test[test_known_class_mask]

        effective_k = min(int(n_neighbors), int(X_train.shape[0]))
        logger.info(
            "{} classification for {} fitting kNN (train_cells={}, test_cells={}, classes={}, k={})",
            stage.capitalize(),
            label_key,
            int(X_train.shape[0]),
            int(len(y_test_known)),
            int(len(train_classes)),
            int(effective_k),
        )
        y_pred, classes, classifier_name = _predict_knn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test_known,
            n_neighbors=effective_k,
            backend=backend,
            log_prefix=f"{stage.capitalize()} classification for {label_key}",
        )
        logger.info("{} classification for {} computing metrics", stage.capitalize(), label_key)

        confusion_matrix_path = None
        if confusion_matrix_dir is not None:
            filename = f"{_safe_filename(stage)}_{_safe_filename(label_key)}.png"
            confusion_matrix_path = Path(confusion_matrix_dir) / filename
            if len(classes) > MAX_CONFUSION_MATRIX_CLASSES:
                logger.info(
                    "{} classification for {} confusion matrix skipped: {} classes exceeds limit {}",
                    stage.capitalize(),
                    label_key,
                    int(len(classes)),
                    MAX_CONFUSION_MATRIX_CLASSES,
                )
                confusion_matrix_path = None
            else:
                logger.info(
                    "{} classification for {} rendering confusion matrix (classes={})",
                    stage.capitalize(),
                    label_key,
                    int(len(classes)),
                )
                _save_confusion_matrix(
                    y_test=y_test_known,
                    y_pred=y_pred,
                    classes=classes,
                    output_path=confusion_matrix_path,
                )
                logger.info(
                    "{} classification for {} confusion matrix saved to {}",
                    stage.capitalize(),
                    label_key,
                    confusion_matrix_path,
                )

        results[label_key] = {
            "accuracy": float(accuracy_score(y_test_known, y_pred)),
            "macro_f1": float(f1_score(y_test_known, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test_known, y_pred, average="weighted", zero_division=0)),
            "count": int(len(y_test_known)),
            "n_classes_train": int(len(train_classes)),
            "n_train": int(X_train.shape[0]),
            "n_test": int(len(y_test_known)),
            "n_train_slides": int(len(train_records)),
            "n_test_slides": int(len(test_records)),
            "n_neighbors": int(effective_k),
            "classifier": classifier_name,
            "classification_backend": backend,
            "split": "train_slide_test_slide",
            "n_test_dropped_unknown_class": int((~test_known_class_mask).sum()),
            "confusion_matrix_path": None if confusion_matrix_path is None else str(confusion_matrix_path),
        }
        logger.info(
            "{} classification for {} complete (accuracy={:.4f}, macro_f1={:.4f}, weighted_f1={:.4f})",
            stage.capitalize(),
            label_key,
            results[label_key]["accuracy"],
            results[label_key]["macro_f1"],
            results[label_key]["weighted_f1"],
        )

    return results


def summarize_split_classification_scores(
    label_keys: list[str] | tuple[str, ...],
    results_by_label_key: dict[str, dict[str, float | int | str] | None],
) -> dict[str, object]:
    metrics_by_label_key = {}
    for label_key in label_keys:
        result = results_by_label_key.get(label_key)
        if result is None:
            metrics_by_label_key[label_key] = {
                "accuracy_mean": None,
                "accuracy_weighted_mean": None,
                "macro_f1_mean": None,
                "macro_f1_weighted_mean": None,
                "weighted_f1_mean": None,
                "weighted_f1_weighted_mean": None,
                "evaluated_slides": 0,
                "split": "train_slide_test_slide",
            }
            continue

        metrics_by_label_key[label_key] = {
            **result,
            "accuracy_mean": result["accuracy"],
            "accuracy_weighted_mean": result["accuracy"],
            "macro_f1_mean": result["macro_f1"],
            "macro_f1_weighted_mean": result["macro_f1"],
            "weighted_f1_mean": result["weighted_f1"],
            "weighted_f1_weighted_mean": result["weighted_f1"],
            "evaluated_slides": result["n_test_slides"],
        }

    return {
        "label_keys": list(label_keys),
        "classification_metrics_by_label_key": metrics_by_label_key,
    }


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
