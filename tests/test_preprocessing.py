
import pytest
import torch
from PIL import Image

from federated.data.preprocessing import (
    get_dataset_config,
    get_transforms,
)


@pytest.mark.parametrize(
    "dataset_name,expected_classes",
    [
        ("fashion_mnist", 10),
        ("pneumonia_mnist", 2),
    ],
)
def test_dataset_metadata(
    dataset_name,
    expected_classes,
):
    config = get_dataset_config(dataset_name)

    assert config["num_classes"] == expected_classes
    assert config["input_channels"] == 1
    assert config["image_size"] == 28


@pytest.mark.parametrize(
    "dataset_name",
    ["fashion_mnist", "pneumonia_mnist"],
)
def test_image_preprocessing(dataset_name):
    image = Image.new(
        mode="L",
        size=(28, 28),
        color=128,
    )

    transform = get_transforms(dataset_name)
    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 28, 28)
    assert torch.isfinite(tensor).all()


def test_unsupported_dataset():
    with pytest.raises(ValueError):
        get_dataset_config("unknown_dataset")
