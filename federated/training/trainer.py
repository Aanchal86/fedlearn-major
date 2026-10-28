
import time

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Train a model for one complete epoch.

    Returns sample-weighted training loss,
    accuracy, number of samples, and duration.
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start_time = time.perf_counter()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty")

    duration = time.perf_counter() - start_time

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_samples": total_samples,
        "duration_seconds": duration,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> list[dict]:
    """
    Train a CNN for multiple epochs.

    Returns a list containing metrics for
    each training epoch.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        metrics["epoch"] = epoch
        history.append(metrics)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Time: {metrics['duration_seconds']:.2f}s"
        )

    return history
