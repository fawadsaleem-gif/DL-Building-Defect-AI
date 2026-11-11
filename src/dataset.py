from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, models

def create_loaders(dataset_root="dataset/BD3_split", batch_size=16, num_workers=0):
    root = Path(dataset_root)
    transform = models.ResNet18_Weights.DEFAULT.transforms()
    datasets_by_split = {
        split: datasets.ImageFolder(root / split, transform=transform)
        for split in ("train", "val", "test")
    }
    class_names = datasets_by_split["train"].classes
    for split, ds in datasets_by_split.items():
        if ds.classes != class_names:
            raise ValueError(f"Class order mismatch in {split}: {ds.classes} != {class_names}")
    loaders = {
        split: DataLoader(ds, batch_size=batch_size,
                          shuffle=(split == "train"), num_workers=num_workers)
        for split, ds in datasets_by_split.items()
    }
    return loaders, datasets_by_split, class_names
