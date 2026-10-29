
from collections import OrderedDict

import torch

from federated.contracts.client_update import ClientUpdate


def federated_averaging(
    updates: list[ClientUpdate],
) -> OrderedDict[str, torch.Tensor]:
    """
    Aggregate client model states using sample-weighted FedAvg.

    Returns a new state dictionary without modifying
    any client update.
    """

    if not updates:
        raise ValueError("At least one client update is required")

    client_ids = set()
    expected_round = updates[0].round_number
    reference_state = updates[0].model_state

    for update in updates:
        update.validate()

        if update.client_id in client_ids:
            raise ValueError("Duplicate client ID")

        client_ids.add(update.client_id)

        if update.round_number != expected_round:
            raise ValueError("Client updates must belong to the same round")

        if update.model_state.keys() != reference_state.keys():
            raise ValueError("Client model parameters do not match")

        for name, tensor in update.model_state.items():
            reference_tensor = reference_state[name]

            if tensor.shape != reference_tensor.shape:
                raise ValueError(f"Shape mismatch for {name}")

            if tensor.dtype != reference_tensor.dtype:
                raise ValueError(f"Dtype mismatch for {name}")

    total_samples = sum(update.num_samples for update in updates)

    aggregated_state = OrderedDict()

    for name, reference_tensor in reference_state.items():
        if reference_tensor.is_floating_point():
            accumulator = torch.zeros_like(
                reference_tensor,
                dtype=torch.float64,
                device="cpu",
            )

            for update in updates:
                weight = update.num_samples / total_samples

                accumulator += (
                    update.model_state[name]
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    * weight
                )

            aggregated_state[name] = accumulator.to(
                dtype=reference_tensor.dtype
            )
        else:
            # Integer/bool buffers cannot be averaged.
            # Preserve them only when all clients agree.
            if not all(
                torch.equal(
                    update.model_state[name].cpu(),
                    reference_tensor.cpu(),
                )
                for update in updates[1:]
            ):
                raise ValueError(
                    f"Non-floating tensor mismatch for {name}"
                )

            aggregated_state[name] = (
                reference_tensor.detach().cpu().clone()
            )

    return aggregated_state
