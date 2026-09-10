import numpy as np
import warnings
from torch.utils.data import Dataset
from typing import Dict, List
from .utils import get_labels


def dirichlet_partition(
    dataset: Dataset,
    n_clients: int,
    alpha: float = 0.5,
    min_samples: int = 10,
    seed: int = 42
) -> Dict[int, List[int]]:
    # alpha -> 0: one class per client (extreme non-IID)
    # alpha = 0.1: high skew, alpha = 0.5: moderate, alpha -> inf: IID
    np.random.seed(seed)

    labels = get_labels(dataset)
    classes = np.unique(labels)
    client_indices = {cid: [] for cid in range(n_clients)}

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)

        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        counts = (proportions * len(cls_idx)).astype(int)
        counts[np.argmax(counts)] += len(cls_idx) - counts.sum()

        ptr = 0
        for cid in range(n_clients):
            client_indices[cid].extend(cls_idx[ptr : ptr + counts[cid]].tolist())
            ptr += counts[cid]

    for cid in range(n_clients):
        np.random.shuffle(client_indices[cid])

    min_count = min(len(v) for v in client_indices.values())
    max_count = max(len(v) for v in client_indices.values())

    if min_count < min_samples:
        warnings.warn(f"Some clients have < {min_samples} samples (min: {min_count}). Try a higher alpha.")

    print(f"[dirichlet] alpha={alpha}, {n_clients} clients, min={min_count}, max={max_count}")
    return client_indices


def compute_emd_proxy(client_indices: Dict[int, List[int]], dataset: Dataset) -> float:
    # Average pairwise L1 distance between client label distributions
    # 0 = IID, ~1 = maximum skew. Used to quantify non-IID level (Zhao et al. 2023)
    labels = get_labels(dataset)
    classes = np.unique(labels)
    n_clients = len(client_indices)

    distributions = []
    for cid in range(n_clients):
        idx = client_indices[cid]
        freq = np.array([np.sum(labels[idx] == cls) for cls in classes], dtype=float)
        freq /= freq.sum() if freq.sum() > 0 else 1.0
        distributions.append(freq)

    total, count = 0.0, 0
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            total += np.sum(np.abs(distributions[i] - distributions[j]))
            count += 1

    emd = total / count if count > 0 else 0.0
    print(f"[emd_proxy] {emd:.4f}  (0=IID, ~1=max skew)")
    return emd
