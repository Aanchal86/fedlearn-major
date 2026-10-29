
from torchvision import transforms


DATASET_CONFIGS = {
    "fashion_mnist": {
        "mean": (0.2860,),
        "std": (0.3530,),
        "num_classes": 10,
        "input_channels": 1,
        "image_size": 28,
    },
    "pneumonia_mnist": {
        "mean": (0.5,),
        "std": (0.5,),
        "num_classes": 2,
        "input_channels": 1,
        "image_size": 28,
    },
}


def get_dataset_config(dataset_name: str) -> dict:
    """
    Return metadata for a supported dataset.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    return DATASET_CONFIGS[dataset_name].copy()


def get_transforms(dataset_name: str):
    """
    Return the image preprocessing pipeline.

    All datasets are converted to tensors and normalized.
    """
    config = get_dataset_config(dataset_name)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["mean"],
            std=config["std"],
        ),
    ])
