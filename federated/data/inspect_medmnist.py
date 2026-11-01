
from collections import Counter
from pathlib import Path

import medmnist
from medmnist import INFO
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    info = INFO["pneumoniamnist"]
    dataset_class = getattr(medmnist, info["python_class"])

    transform = transforms.ToTensor()

    train_dataset = dataset_class(
        split="train",
        root=str(DATA_DIR),
        transform=transform,
        download=True,
        size=28,
    )

    val_dataset = dataset_class(
        split="val",
        root=str(DATA_DIR),
        transform=transform,
        download=True,
        size=28,
    )

    test_dataset = dataset_class(
        split="test",
        root=str(DATA_DIR),
        transform=transform,
        download=True,
        size=28,
    )

    print("\n===== PneumoniaMNIST =====")
    print("Task:", info["task"])
    print("Channels:", info["n_channels"])
    print("Classes:", info["label"])

    print("\n===== Dataset Splits =====")
    print("Training:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Testing:", len(test_dataset))

    image, label = train_dataset[0]

    print("\n===== Sample =====")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Pixel range:", image.min().item(), image.max().item())
    print("Label:", label)

    print("\n===== Class Distribution =====")

    for split_name, dataset in [
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ]:
        counts = Counter(dataset.labels.flatten().tolist())

        print(f"\n{split_name.upper()}:")
        for class_id, class_name in info["label"].items():
            print(
                f"{class_id}: {class_name} "
                f"-> {counts[int(class_id)]}"
            )

    print("\nMedMNIST inspection completed!")


if __name__ == "__main__":
    main()
