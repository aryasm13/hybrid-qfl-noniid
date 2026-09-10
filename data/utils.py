import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from typing import Tuple, List
import matplotlib.pyplot as plt

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_dataset(name: str = "fashionmnist", normalize: bool = True) -> Tuple[Dataset, Dataset]:
    os.makedirs(DATA_ROOT, exist_ok=True)
    name = name.lower().strip()

    if name == "fashionmnist":
        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize((0.5,), (0.5,)))
        transform = transforms.Compose(t)
        train = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    elif name == "pathmnist":
        try:
            from medmnist import PathMNIST
        except ImportError:
            raise ImportError("Run: pip install medmnist")
        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        transform = transforms.Compose(t)
        train = PathMNIST(split="train", transform=transform, download=True, root=DATA_ROOT)
        test = PathMNIST(split="test", transform=transform, download=True, root=DATA_ROOT)

    else:
        raise ValueError(f"Unknown dataset '{name}'. Use 'fashionmnist' or 'pathmnist'.")

    print(f"[data] {name}: {len(train)} train / {len(test)} test")
    return train, test


def get_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
    elif hasattr(dataset, "labels"):
        return np.array(dataset.labels).flatten()
    return np.array([dataset[i][1] for i in range(len(dataset))])


def visualize_partition(client_indices: dict, dataset: Dataset, title: str = "Client Distribution", save_path: str = None):
    labels = get_labels(dataset)
    n_clients = len(client_indices)
    n_classes = len(np.unique(labels))

    counts = np.zeros((n_clients, n_classes), dtype=int)
    for cid, idx in client_indices.items():
        client_labels = labels[idx]
        for cls in range(n_classes):
            counts[cid, cls] = np.sum(client_labels == cls)

    fig, ax = plt.subplots(figsize=(max(10, n_clients * 0.6), 5))
    bottom = np.zeros(n_clients)
    for cls in range(n_classes):
        ax.bar(range(n_clients), counts[:, cls], bottom=bottom, label=f"Class {cls}")
        bottom += counts[:, cls]

    ax.set_title(title)
    ax.set_xlabel("Client")
    ax.set_ylabel("Sample count")
    ax.legend(loc="upper right", ncol=5, fontsize=7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[viz] saved to {save_path}")

    plt.show()
