"""Classification metrics on predicted class probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

BINARY_CLASS_COUNT = 2


@dataclass(frozen=True)
class ClassificationMetrics:
    """Test-set metrics of one classifier."""

    accuracy: float
    balanced_accuracy: float
    auroc: float


def compute_classification_metrics(*, labels: np.ndarray, probabilities: np.ndarray) -> ClassificationMetrics:
    """Return accuracy, balanced accuracy, and AUROC (macro one-vs-rest beyond two classes).

    Args:
        labels: ``(N,)`` integer class labels.
        probabilities: ``(N, C)`` class probabilities.
    """
    predictions = probabilities.argmax(axis=1)
    if probabilities.shape[1] == BINARY_CLASS_COUNT:
        auroc = roc_auc_score(labels, probabilities[:, 1])
    else:
        auroc = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        auroc=float(auroc),
    )
