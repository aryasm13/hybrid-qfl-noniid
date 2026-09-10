import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List


def quantity_skew_partition(
    dataset: Dataset,
    n_clients: int,
    sigma: float = 1.0,
    min_samples: int = 20,
    seed: int = 42
) -> Dict[int, List[int]]:
    # sigma = 0: equal sizes, sigma = 1.0: high skew, sigma = 2.0: extreme skew
    np.random.seed(seed)

    n_samples = len(dataset)
    raw = np.random.lognormal(mean=0.0, sigma=sigma, size=n_clients)
    proportions = raw / raw.sum()
    sizes = (proportions * n_samples).astype(int)
    sizes = np.maximum(sizes, min_samples)

    while sizes.sum() > n_samples:
        excess = sizes.sum() - n_samples
        biggest = np.argmax(sizes - min_samples)
        cut = min(excess, sizes[biggest] - min_samples)
        sizes[biggest] -= cut

    shuffled = np.random.permutation(n_samples)
    client_indices = {}
    ptr = 0
    for cid in range(n_clients):
        client_indices[cid] = shuffled[ptr : ptr + sizes[cid]].tolist()
        ptr += sizes[cid]

    counts = [len(v) for v in client_indices.values()]
    print(f"[quantity_skew] sigma={sigma}, {n_clients} clients, min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.0f}")
    return client_indices
