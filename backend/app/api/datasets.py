
from fastapi import APIRouter, HTTPException

from federated.data.preprocessing import DATASET_CONFIGS


router = APIRouter(
    prefix="/api/datasets",
    tags=["Datasets"],
)


DATASET_DETAILS = {
    "fashion_mnist": {
        "display_name": "Fashion-MNIST",
        "description": "10-class clothing image classification",
        "train_samples": 54000,
        "val_samples": 6000,
        "test_samples": 10000,
    },
    "pneumonia_mnist": {
        "display_name": "PneumoniaMNIST",
        "description": "Binary chest X-ray image classification",
        "train_samples": 4708,
        "val_samples": 524,
        "test_samples": 624,
    },
}


def build_dataset_info(dataset_name: str) -> dict:
    config = DATASET_CONFIGS[dataset_name]
    details = DATASET_DETAILS[dataset_name]

    return {
        "name": dataset_name,
        **details,
        "num_classes": config["num_classes"],
        "input_channels": config["input_channels"],
        "image_size": config["image_size"],
    }


@router.get("")
def list_datasets():
    """Return metadata for all supported datasets."""

    return [
        build_dataset_info(name)
        for name in DATASET_DETAILS
    ]


@router.get("/{dataset_name}")
def get_dataset(dataset_name: str):
    """Return metadata for a specific dataset."""

    if dataset_name not in DATASET_DETAILS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return build_dataset_info(dataset_name)
