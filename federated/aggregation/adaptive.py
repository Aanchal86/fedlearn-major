
from collections import OrderedDict
import math

import torch

from federated.contracts.client_update import ClientUpdate
from federated.aggregation.fedavg import federated_averaging


def calculate_adaptive_weights(
    updates: list[ClientUpdate],
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Calculate sample-size and accuracy-based weights."""

    if not updates:
        raise ValueError("At least one client update is required")

    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0, 1]")

    # Reuse FedAvg's validation of update compatibility.
    federated_averaging(updates)

    raw_weights = {}

    for update in updates:
        accuracy = float(update.training_accuracy)

        if not math.isfinite(accuracy):
            raise ValueError("Training accuracy must be finite")

        if not 0 <= accuracy <= 1:
            raise ValueError(
                "Training accuracy must be between 0 and 1"
            )

        raw_weights[update.client_id] = (
            update.num_samples * max(accuracy, epsilon)
        )

    total = sum(raw_weights.values())

    return {
        client_id: weight / total
        for client_id, weight in raw_weights.items()
    }


def adaptive_averaging(
    updates: list[ClientUpdate],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float]]:
    """Aggregate model parameters using adaptive weights."""

    weights = calculate_adaptive_weights(updates)
    reference_state = updates[0].model_state

    aggregated_state = OrderedDict()

    for name, reference_tensor in reference_state.items():
        if reference_tensor.is_floating_point():
            accumulator = torch.zeros_like(
                reference_tensor,
                dtype=torch.float64,
                device="cpu",
            )

            for update in updates:
                accumulator += (
                    update.model_state[name]
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    * weights[update.client_id]
                )

            aggregated_state[name] = accumulator.to(
                dtype=reference_tensor.dtype
            )
        else:
            # Compatibility already checked by FedAvg.
            aggregated_state[name] = (
                reference_tensor.detach().cpu().clone()
            )

    return aggregated_state, weights
