import numpy as np


def aggregate_weighted(client_weights, sample_counts, label_distributions, lam=0.5):
    n_clients = len(client_weights)
    total_samples = sum(sample_counts)

    # size-based weight (same as FedAvg)
    size_w = np.array([c / total_samples for c in sample_counts])

    # entropy-based weight: clients with more balanced labels get higher weight
    entropies = []
    for dist in label_distributions:
        p = np.array(dist)
        p = p[p > 0]
        entropies.append(-np.sum(p * np.log(p)))
    entropies = np.array(entropies)
    entropy_w = entropies / (entropies.sum() + 1e-9)

    weights = (1 - lam) * size_w + lam * entropy_w
    weights = weights / weights.sum()

    global_weights = {}
    for key in client_weights[0].keys():
        global_weights[key] = sum(
            weights[i] * client_weights[i][key].float()
            for i in range(n_clients)
        )
    return global_weights
