import torchmetrics
import torch

def compute_metrics(num_classes=31, device=None):
    metrics = torchmetrics.MetricCollection({
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro"),
        "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro"),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        "precision_weighted": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="weighted"),
        "recall_weighted": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="weighted"),
        "f1_weighted": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="weighted")
    })
    if device is not None:
        return metrics.to(device)
    return metrics