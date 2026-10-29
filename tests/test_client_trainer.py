
import copy

import pytest
import torch
from torch.utils.data import TensorDataset

from federated.client.trainer import ClientTrainer
from federated.contracts.client_update import ClientUpdate
from federated.models.cnn import SmallCNN


@pytest.fixture
def sample_dataset():
    torch.manual_seed(42)
    images = torch.randn(20, 1, 28, 28)
    labels = torch.randint(0, 2, (20,))
    return TensorDataset(images, labels)


def test_client_returns_valid_update(sample_dataset):
    model = SmallCNN(num_classes=2)
    client = ClientTrainer(
        client_id="client_1",
        dataset=sample_dataset,
        batch_size=5,
    )

    update = client.train(
        global_model=model,
        round_number=1,
        local_epochs=1,
    )

    assert isinstance(update, ClientUpdate)
    assert update.client_id == "client_1"
    assert update.round_number == 1
    assert update.num_samples == 20
    assert update.training_loss >= 0
    assert 0 <= update.training_accuracy <= 1
    assert update.training_time_seconds >= 0
    assert update.model_state

    update.validate()


def test_global_model_is_unchanged(sample_dataset):
    model = SmallCNN(num_classes=2)
    original_state = copy.deepcopy(model.state_dict())

    client = ClientTrainer("client_1", sample_dataset)
    client.train(model, round_number=1)

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original_state[name])


def test_local_training_changes_weights(sample_dataset):
    model = SmallCNN(num_classes=2)
    original_state = copy.deepcopy(model.state_dict())

    client = ClientTrainer("client_1", sample_dataset)
    update = client.train(model, round_number=1)

    assert any(
        not torch.equal(update.model_state[name], original_state[name])
        for name in original_state
    )


def test_clients_train_independently(sample_dataset):
    model = SmallCNN(num_classes=2)
    original_state = copy.deepcopy(model.state_dict())

    client_1 = ClientTrainer("client_1", sample_dataset, seed=42)
    client_2 = ClientTrainer("client_2", sample_dataset, seed=123)

    update_1 = client_1.train(model, round_number=1)
    update_2 = client_2.train(model, round_number=1)

    assert update_1.client_id != update_2.client_id
    assert update_1.model_state is not update_2.model_state

    for name in original_state:
        assert torch.equal(model.state_dict()[name], original_state[name])


def test_invalid_client_configuration(sample_dataset):
    with pytest.raises(ValueError):
        ClientTrainer("", sample_dataset)

    with pytest.raises(ValueError):
        ClientTrainer("client_1", sample_dataset, batch_size=0)

    with pytest.raises(ValueError):
        ClientTrainer("client_1", TensorDataset(
            torch.empty(0, 1, 28, 28),
            torch.empty(0, dtype=torch.long),
        ))


def test_invalid_training_configuration(sample_dataset):
    model = SmallCNN(num_classes=2)
    client = ClientTrainer("client_1", sample_dataset)

    with pytest.raises(ValueError):
        client.train(model, round_number=0)

    with pytest.raises(ValueError):
        client.train(model, round_number=1, local_epochs=0)

    with pytest.raises(ValueError):
        client.train(model, round_number=1, learning_rate=0)
