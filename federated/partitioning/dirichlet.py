
import numpy as np
from torch.utils.data import Dataset, Subset


def partition_dirichlet(
    dataset: Dataset,
    num_clients: int = 5,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples: int = 1,
) -> dict[int, Subset]:
    """
    Partition training samples across clients using
    a Dirichlet label-skew distribution.

    Smaller alpha -> stronger class imbalance.
    Larger alpha -> more balanced label proportions.
    """

    if num_clients < 1:
        raise ValueError("num_clients must be positive")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    if len(dataset) < num_clients * min_samples:
        raise ValueError(
            "Not enough samples for the requested minimum"
        )

    rng = np.random.default_rng(seed)

    labels = np.array([
        int(np.asarray(dataset[i][1]).item())
        for i in range(len(dataset))
    ])

    classes = np.unique(labels)

    # Retry until each client receives enough samples.
    for attempt in range(1000):
        client_indices = [[] for _ in range(num_clients)]

        for label in classes:
            class_indices = np.flatnonzero(labels == label)
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.full(num_clients, alpha)
            )

            # Multinomial allocation preserves every sample.
            counts = rng.multinomial(
                len(class_indices),
                proportions,
            )

            start = 0
            for client_id, count in enumerate(counts):
                end = start + count
                client_indices[client_id].extend(
                    class_indices[start:end].tolist()
                )
                start = end

        if all(
            len(indices) >= min_samples
            for indices in client_indices
        ):
            return {
                client_id: Subset(
                    dataset,
                    rng.permutation(indices).tolist(),
                )
                for client_id, indices in enumerate(client_indices)
            }

    raise RuntimeError(
        "Could not create valid Dirichlet partitions "
        "after 1000 attempts. Try increasing alpha "
        "or reducing min_samples."
    )
