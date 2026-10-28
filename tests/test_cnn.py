
import pytest
import torch

from federated.models.cnn import SmallCNN, create_model


@pytest.mark.parametrize(
    "dataset_name, expected_classes",
    [
        ("fashion_mnist", 10),
        ("pneumonia_mnist", 2),
    ],
)
def test_model_output_shape(dataset_name, expected_classes):
    model = create_model(dataset_name)

    images = torch.randn(8, 1, 28, 28)

    outputs = model(images)

    assert outputs.shape == (8, expected_classes)


def test_model_supports_single_image():
    model = create_model("fashion_mnist")

    image = torch.randn(1, 1, 28, 28)

    output = model(image)

    assert output.shape == (1, 10)


def test_model_outputs_finite_values():
    model = create_model("pneumonia_mnist")

    images = torch.randn(4, 1, 28, 28)

    outputs = model(images)

    assert torch.isfinite(outputs).all()


def test_invalid_num_classes():
    with pytest.raises(ValueError):
        SmallCNN(num_classes=1)


def test_invalid_input_channels():
    with pytest.raises(ValueError):
        SmallCNN(num_classes=10, input_channels=0)


def test_unsupported_dataset():
    with pytest.raises((ValueError, KeyError)):
        create_model("unknown_dataset")


def test_model_gradient_flow():
    model = create_model("fashion_mnist")

    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])

    criterion = torch.nn.CrossEntropyLoss()

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_model_parameters_are_trainable():
    model = create_model("fashion_mnist")

    parameters = list(model.parameters())

    assert len(parameters) > 0
    assert all(parameter.requires_grad for parameter in parameters)


def test_model_parameter_count():
    fashion_model = create_model("fashion_mnist")
    pneumonia_model = create_model("pneumonia_mnist")

    fashion_parameters = sum(
        parameter.numel()
        for parameter in fashion_model.parameters()
    )

    pneumonia_parameters = sum(
        parameter.numel()
        for parameter in pneumonia_model.parameters()
    )

    assert fashion_parameters == 206922
    assert pneumonia_parameters == 205890
