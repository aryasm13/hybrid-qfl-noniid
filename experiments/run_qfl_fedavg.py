import os
import sys
import copy
import argparse
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_dataset, iid_partition, dirichlet_partition, quantity_skew_partition, compute_emd_proxy, DATASETS
from models import HybridQNN
from federation import train_one_round, aggregate_fedavg
from evaluation import evaluate, client_drift


def run(num_clients=10, rounds=5, local_epochs=1, batch_size=32, lr=0.001,
        partition="dirichlet", alpha=0.5, seed=42, fixed_readout=False, twin=False,
        frozen_readout=False, pauli_readout=False, random_codes=False, dataset="fashionmnist", folder="",
        device="cpu"):

    if twin:
        name = "qfl_twin_fco" if fixed_readout else "qfl_twin"
    elif pauli_readout:
        name = "qfl_pauli"
    elif frozen_readout:
        name = "qfl_frozen"
    elif random_codes:
        name = "qfl_randcode"
    else:
        name = "qfl_fco" if fixed_readout else "qfl_fedavg"
    print(f"\n=== QFL FedAvg | {name} | {dataset} | partition={partition} | alpha={alpha} | clients={num_clients} ===\n")

    train_data, test_data = load_dataset(dataset)
    in_channels, num_classes = DATASETS[dataset]
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

    torch.manual_seed(seed)
    global_model = HybridQNN(in_channels=in_channels, num_classes=num_classes,
                             fixed_readout=fixed_readout, twin=twin,
                             frozen_readout=frozen_readout, pauli_readout=pauli_readout,
                             random_codes=random_codes)
    print(f"Model parameters: {global_model.count_parameters():,}\n")

    history = []
    for r in range(1, rounds + 1):
        start = time.time()
        client_weights = []
        for cid in range(num_clients):
            local_model = copy.deepcopy(global_model)
            weights = train_one_round(local_model, client_loaders[cid],
                                      epochs=local_epochs, lr=lr, device=device)
            client_weights.append(weights)

        drift = client_drift(global_model.state_dict(), client_weights, sample_counts)
        global_model.load_state_dict(aggregate_fedavg(client_weights, sample_counts))
        metrics = evaluate(global_model, test_loader, device=device)
        elapsed = time.time() - start
        print(f"Round {r:02d}/{rounds} | loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f} | time={elapsed:.1f}s")
        history.append({"round": r, "seed": seed, "emd": emd, **metrics, **drift, "time": round(elapsed, 1)})

    out_dir = os.path.join("results", "tables", folder)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}_{partition}_a{alpha}_c{num_clients}_e{local_epochs}_lr{lr}_s{seed}.csv")
    pd.DataFrame(history).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients",    type=int,   default=10)
    parser.add_argument("--rounds",     type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--partition",  type=str,   default="dirichlet", choices=["iid", "dirichlet", "quantity"])
    parser.add_argument("--alpha",      type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--dataset",    type=str,   default="fashionmnist", choices=list(DATASETS))
    parser.add_argument("--folder",     type=str,   default="")
    parser.add_argument("--fixed_readout", action="store_true")
    parser.add_argument("--twin",          action="store_true")
    parser.add_argument("--frozen_readout", action="store_true")
    parser.add_argument("--pauli_readout",  action="store_true")
    parser.add_argument("--random_codes",   action="store_true")
    args = parser.parse_args()

    run(num_clients=args.clients, rounds=args.rounds, local_epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        partition=args.partition, alpha=args.alpha, seed=args.seed,
        fixed_readout=args.fixed_readout, twin=args.twin,
        frozen_readout=args.frozen_readout, pauli_readout=args.pauli_readout,
        random_codes=args.random_codes, dataset=args.dataset, folder=args.folder)
