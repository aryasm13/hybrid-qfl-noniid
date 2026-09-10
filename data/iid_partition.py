import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List
from .utils import get_labels


def iid_partition(dataset: Dataset, n_clients: int, seed: int = 42) -> Dict[int, List[int]]:
    np.random.seed(seed)

    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)

    samples_per_client = n_samples // n_clients
    client_indices = {}
    for cid in range(n_clients):
        start = cid * samples_per_client
        client_indices[cid] = indices[start : start + samples_per_client].tolist()

    labels = get_labels(dataset)
    print(f"[iid] {n_clients} clients, {samples_per_client} samples each")
    _print_summary(client_indices, labels)
    return client_indices


def _print_summary(client_indices, labels):
    print(f"{'Client':<10} {'Samples':<10} {'Classes'}")
    for cid, idx in list(client_indices.items())[:5]:
        classes = sorted(np.unique(labels[idx]).tolist())
        print(f"  C{cid:<7} {len(idx):<10} {classes}")
    if len(client_indices) > 5:
        print(f"  ... {len(client_indices) - 5} more clients")
