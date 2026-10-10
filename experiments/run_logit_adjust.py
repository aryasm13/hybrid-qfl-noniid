import os
import sys
import copy
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_dataset, iid_partition, dirichlet_partition, quantity_skew_partition, get_labels, compute_emd_proxy
from models import HybridQNN
from federation import train_logit_adjusted, aggregate_fedavg
from evaluation import evaluate, client_drift


def get_log_prior(dataset, indices, n_classes=10):
    counts = np.bincount(get_labels(dataset)[indices], minlength=n_classes)
    return torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32) + 1e-4)


def run(num_clients=10, rounds=5, local_epochs=1, batch_size=32, lr=0.001,
        tau=1.0, partition="dirichlet", alpha=0.5, seed=42, device="cpu"):

    print(f"\n=== Logit-adjusted FedAvg | tau={tau} | partition={partition} | alpha={alpha} | clients={num_clients} ===\n")

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
    emd = compute_emd_proxy(client_indices, train_data)
    log_priors = [get_log_prior(train_data, client_indices[cid]) for cid in range(num_clients)]

    torch.manual_seed(seed)
    global_model = HybridQNN(in_channels=1, num_classes=10)
    print(f"Model parameters: {global_model.count_parameters():,}\n")

    history = []
    for r in range(1, rounds + 1):
        start = time.time()
        client_weights = []
        for cid in range(num_clients):
            local_model = copy.deepcopy(global_model)
            weights = train_logit_adjusted(local_model, client_loaders[cid], log_priors[cid], tau=tau,
                                           epochs=local_epochs, lr=lr, device=device)
            client_weights.append(weights)

        drift = client_drift(global_model.state_dict(), client_weights, sample_counts)
        global_model.load_state_dict(aggregate_fedavg(client_weights, sample_counts))
        metrics = evaluate(global_model, test_loader, device=device)
        elapsed = time.time() - start
        print(f"Round {r:02d}/{rounds} | loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f} | time={elapsed:.1f}s")
        history.append({"round": r, "tau": tau, "seed": seed, "emd": emd, **metrics, **drift, "time": round(elapsed, 1)})

    os.makedirs("results/tables", exist_ok=True)
    out = f"results/tables/qfl_logitadj_{partition}_a{alpha}_tau{tau}_c{num_clients}_e{local_epochs}_lr{lr}_s{seed}.csv"
    pd.DataFrame(history).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients",    type=int,   default=10)
    parser.add_argument("--rounds",     type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--tau",        type=float, default=1.0)
    parser.add_argument("--partition",  type=str,   default="dirichlet", choices=["iid", "dirichlet", "quantity"])
    parser.add_argument("--alpha",      type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    run(num_clients=args.clients, rounds=args.rounds, local_epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, tau=args.tau,
        partition=args.partition, alpha=args.alpha, seed=args.seed)
