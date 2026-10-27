
from federated.data.splits import get_dataset_splits


def test_fashion_split_sizes():
    splits = get_dataset_splits("fashion_mnist")

    assert len(splits.train) == 54000
    assert len(splits.val) == 6000
    assert len(splits.test) == 10000


def test_fashion_split_reproducibility():
    first = get_dataset_splits(
        "fashion_mnist", seed=42
    )

    second = get_dataset_splits(
        "fashion_mnist", seed=42
    )

    assert first.train.indices == second.train.indices
    assert first.val.indices == second.val.indices


def test_fashion_no_overlap():
    splits = get_dataset_splits("fashion_mnist")

    train_indices = set(splits.train.indices)
    val_indices = set(splits.val.indices)

    assert train_indices.isdisjoint(val_indices)

    assert len(train_indices | val_indices) == 60000


def test_pneumonia_split_sizes():
    splits = get_dataset_splits("pneumonia_mnist")

    assert len(splits.train) == 4708
    assert len(splits.val) == 524
    assert len(splits.test) == 624
