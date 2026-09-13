import os
import sys
import copy
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_dataset, iid_partition, dirichlet_partition, quantity_skew_partition
from models import ClassicalCNN
from federation import train_one_round, aggregate_fedavg
from evaluation import evaluate


def run(num_clients=10, rounds=5, local_epochs=1, batch_size=32, lr=0.001,
        partition="dirichlet", alpha=0.5, device="cpu"):

    print(f"\n=== Classical FL Baseline | partition={partition} | alpha={alpha} | clients={num_clients} ===\n")

    train_data, test_data = load_dataset("fashionmnist")
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

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

    global_model = ClassicalCNN(in_channels=1, num_classes=10)
    print(f"Model parameters: {global_model.count_parameters():,}\n")

    history = []
    for r in range(1, rounds + 1):
        client_weights = []
        for cid in range(num_clients):
            local_model = copy.deepcopy(global_model)
            weights = train_one_round(local_model, client_loaders[cid],
                                      epochs=local_epochs, lr=lr, device=device)
            client_weights.append(weights)

        global_model.load_state_dict(aggregate_fedavg(client_weights, sample_counts))
        metrics = evaluate(global_model, test_loader, device=device)
        print(f"Round {r:02d}/{rounds} | loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f}")
        history.append({"round": r, **metrics})

    os.makedirs("results/tables", exist_ok=True)
    out = f"results/tables/classical_{partition}_a{alpha}_c{num_clients}.csv"
    pd.DataFrame(history).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients",   type=int,   default=10)
    parser.add_argument("--rounds",    type=int,   default=5)
    parser.add_argument("--epochs",    type=int,   default=1)
    parser.add_argument("--lr",        type=float, default=0.001)
    parser.add_argument("--partition", type=str,   default="dirichlet", choices=["iid", "dirichlet", "quantity"])
    parser.add_argument("--alpha",     type=float, default=0.5)
    args = parser.parse_args()

    run(num_clients=args.clients, rounds=args.rounds, local_epochs=args.epochs,
        lr=args.lr, partition=args.partition, alpha=args.alpha)
