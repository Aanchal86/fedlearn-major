
from collections import Counter
from pathlib import Path

from torchvision import datasets, transforms


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    # Convert images to PyTorch tensors
    transform = transforms.ToTensor()

    # Download training dataset
    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    # Download test dataset
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    print("\n===== Fashion-MNIST Dataset =====")

    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    # Inspect one image
    image, label = train_dataset[0]

    print("\n===== Sample Image =====")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Pixel minimum:", image.min().item())
    print("Pixel maximum:", image.max().item())
    print("Class index:", label)
    print("Class name:", train_dataset.classes[label])

    # Class distribution
    print("\n===== Training Class Distribution =====")

    counts = Counter(train_dataset.targets.tolist())

    for class_index, class_name in enumerate(
        train_dataset.classes
    ):
        print(
            f"{class_index}: {class_name} "
            f"-> {counts[class_index]} samples"
        )

    print("\nDataset inspection completed!")


if __name__ == "__main__":
    main()
