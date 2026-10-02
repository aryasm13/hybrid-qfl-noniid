import os
import sys
import copy
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_dataset, iid_partition, dirichlet_partition, quantity_skew_partition
from models import HybridQNN
from federation import train_fedprox, aggregate_fedavg
from evaluation import evaluate


def run(num_clients=10, rounds=5, local_epochs=1, batch_size=32, lr=0.001,
        mu=0.01, quantum_only=True, partition="dirichlet", alpha=0.5, seed=42, device="cpu"):

    print(f"\n=== Quantum FedProx | mu={mu} | quantum_only={quantum_only} | partition={partition} | alpha={alpha} ===\n")

    train_data, test_data = load_dataset("fashionmnist")
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    if partition == "iid":
        client_indices = iid_partition(train_data, n_clients=num_clients, seed=seed)
    elif partition == "dirichlet":
        client_indices = dirichlet_partition(train_data, n_clients=num_clients, alpha=alpha, seed=seed)
    else:
        client_indices = quantity_skew_partition(train_data, n_clients=num_clients, seed=seed)

    client_loaders = [
        DataLoader(Subset(train_data, client_indices[cid]), batch_size=batch_size, shuffle=True)
        for cid in range(num_clients)
    ]
    sample_counts = [len(client_indices[cid]) for cid in range(num_clients)]

    torch.manual_seed(seed)
    global_model = HybridQNN(in_channels=1, num_classes=10)
    print(f"Model parameters: {global_model.count_parameters():,}\n")

    history = []
    for r in range(1, rounds + 1):
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        client_weights = []

        for cid in range(num_clients):
            local_model = copy.deepcopy(global_model)
            weights = train_fedprox(local_model, global_state, client_loaders[cid],
                                    mu=mu, epochs=local_epochs, lr=lr,
                                    quantum_only=quantum_only, device=device)
            client_weights.append(weights)

        global_model.load_state_dict(aggregate_fedavg(client_weights, sample_counts))
        metrics = evaluate(global_model, test_loader, device=device)
        print(f"Round {r:02d}/{rounds} | loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f}")
        history.append({"round": r, "mu": mu, "quantum_only": quantum_only, **metrics})

    os.makedirs("results/tables", exist_ok=True)
    out = f"results/tables/qfl_fedprox_{partition}_a{alpha}_mu{mu}_c{num_clients}.csv"
    pd.DataFrame(history).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients",      type=int,   default=10)
    parser.add_argument("--rounds",       type=int,   default=5)
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--mu",           type=float, default=0.01)
    parser.add_argument("--quantum_only", action="store_true", default=True)
    parser.add_argument("--partition",    type=str,   default="dirichlet", choices=["iid", "dirichlet", "quantity"])
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    run(num_clients=args.clients, rounds=args.rounds, local_epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, mu=args.mu, quantum_only=args.quantum_only,
        partition=args.partition, alpha=args.alpha, seed=args.seed)
