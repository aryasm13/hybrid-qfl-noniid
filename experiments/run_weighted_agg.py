import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_dataset, iid_partition, dirichlet_partition, quantity_skew_partition, get_labels
from models import HybridQNN
from federation import train_one_round, aggregate_weighted
from evaluation import evaluate


def get_label_distribution(dataset, indices, n_classes=10):
    labels = get_labels(dataset)
    client_labels = labels[indices]
    counts = np.bincount(client_labels, minlength=n_classes).astype(float)
    counts /= counts.sum() + 1e-9
    return counts.tolist()


def run(num_clients=10, rounds=5, local_epochs=1, batch_size=32, lr=0.001,
        lam=0.5, partition="dirichlet", alpha=0.5, device="cpu"):

    print(f"\n=== Weighted Aggregation | lam={lam} | partition={partition} | alpha={alpha} | clients={num_clients} ===\n")

    train_data, test_data = load_dataset("fashionmnist")
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    if partition == "iid":
        client_indices = iid_partition(train_data, n_clients=num_clients)
    elif partition == "dirichlet":
        client_indices = dirichlet_partition(train_data, n_clients=num_clients, alpha=alpha)
    else:
        client_indices = quantity_skew_partition(train_data, n_clients=num_clients)

    client_loaders = [
        DataLoader(Subset(train_data, client_indices[cid]), batch_size=batch_size, shuffle=True)
        for cid in range(num_clients)
    ]
    sample_counts = [len(client_indices[cid]) for cid in range(num_clients)]
    label_dists = [
        get_label_distribution(train_data, client_indices[cid])
        for cid in range(num_clients)
    ]

    global_model = HybridQNN(in_channels=1, num_classes=10)
    print(f"Model parameters: {global_model.count_parameters():,}\n")

    history = []
    for r in range(1, rounds + 1):
        client_weights = []
        for cid in range(num_clients):
            local_model = copy.deepcopy(global_model)
            weights = train_one_round(local_model, client_loaders[cid],
                                      epochs=local_epochs, lr=lr, device=device)
            client_weights.append(weights)

        global_model.load_state_dict(
            aggregate_weighted(client_weights, sample_counts, label_dists, lam=lam)
        )
        metrics = evaluate(global_model, test_loader, device=device)
        print(f"Round {r:02d}/{rounds} | loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f}")
        history.append({"round": r, "lam": lam, **metrics})

    os.makedirs("results/tables", exist_ok=True)
    out = f"results/tables/qfl_weighted_{partition}_a{alpha}_lam{lam}_c{num_clients}.csv"
    pd.DataFrame(history).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients",    type=int,   default=10)
    parser.add_argument("--rounds",     type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--lam",        type=float, default=0.5)
    parser.add_argument("--partition",  type=str,   default="dirichlet", choices=["iid", "dirichlet", "quantity"])
    parser.add_argument("--alpha",      type=float, default=0.5)
    args = parser.parse_args()

    run(num_clients=args.clients, rounds=args.rounds, local_epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, lam=args.lam,
        partition=args.partition, alpha=args.alpha)
