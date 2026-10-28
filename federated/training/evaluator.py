
import time

import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """
    Evaluate a classification model without updating weights.

    Returns loss, accuracy, macro/weighted metrics,
    per-class metrics, confusion matrix, and duration.
    """

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    start_time = time.perf_counter()

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(
                predictions.cpu().tolist()
            )

    if total_samples == 0:
        raise ValueError("Evaluation DataLoader is empty")

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_predictions, dtype=int)

    num_classes = model.classifier[-1].out_features
    class_ids = list(range(num_classes))

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_ids,
            zero_division=0,
        )
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    weighted_f1 = float(
        np.average(f1, weights=support)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids,
    )

    duration = time.perf_counter() - start_time

    return {
        "loss": total_loss / total_samples,
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
        "num_samples": total_samples,
        "duration_seconds": duration,
    }
