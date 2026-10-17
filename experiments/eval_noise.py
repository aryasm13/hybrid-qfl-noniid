import os
import sys
import time
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_dataset, DATASETS
from models import HybridQNN
from evaluation import noise_accuracies
from eval_shots import KINDS, NAME


def run(dataset="fashionmnist", folder=None, gate_levels=(0.001, 0.005, 0.01, 0.02, 0.05),
        readout_levels=(0.01, 0.02, 0.05, 0.1)):
    if folder is None:
        folder = "saved" if dataset == "fashionmnist" else dataset
    gate_levels = sorted({0.0, *gate_levels})
    readout_levels = sorted({0.0, *readout_levels})
    model_dir = os.path.join("results", "models", folder)
    _, test_data = load_dataset(dataset)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    in_channels, num_classes = DATASETS[dataset]

    rows = []
    for fname in sorted(os.listdir(model_dir)):
        match = NAME.match(fname)
        if not match:
            continue
        kind, partition, alpha, clients, run_seed = match.groups()
        start = time.time()
        model = HybridQNN(in_channels=in_channels, num_classes=num_classes, **KINDS[kind])
        model.load_state_dict(torch.load(os.path.join(model_dir, fname)))
        accuracy, fidelity = noise_accuracies(model, test_loader, gate_levels, readout_levels,
                                              pairs=(kind == "qfl_pauli"))
        for (p, q), acc in accuracy.items():
            rows.append({"kind": kind, "partition": partition, "alpha": float(alpha), "clients": int(clients),
                         "seed": int(run_seed), "gate_noise": p, "readout_error": q,
                         "accuracy": acc, "fidelity": fidelity[p]})
        print(f"{fname[:-3]}: exact {accuracy[(0.0, 0.0)]:.4f} | "
              + " | ".join(f"p {p} {accuracy[(p, 0.0)]:.4f} (fidelity {fidelity[p]:.3f})" for p in gate_levels[1:])
              + " | " + " | ".join(f"q {q} {accuracy[(0.0, q)]:.4f}" for q in readout_levels[1:])
              + f" | {time.time() - start:.0f}s", flush=True)

    out = os.path.join("results", "tables", folder, "noise.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,   default="fashionmnist", choices=list(DATASETS))
    parser.add_argument("--folder",  type=str,   default=None)
    parser.add_argument("--gate",    type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--readout", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    args = parser.parse_args()

    run(dataset=args.dataset, folder=args.folder, gate_levels=args.gate, readout_levels=args.readout)
