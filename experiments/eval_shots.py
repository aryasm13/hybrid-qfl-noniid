import os
import re
import sys
import time
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import load_dataset, DATASETS
from models import HybridQNN
from evaluation import evaluate, shot_accuracies

# hybrid models only: the twins and the classical CNN have no quantum measurement
KINDS = {
    "qfl_fedavg": {}, "qfl_fedprox_full": {}, "qfl_fedprox": {}, "qfl_weighted": {}, "qfl_logitadj": {},
    "qfl_fco": {"fixed_readout": True}, "qfl_frozen": {"frozen_readout": True},
    "qfl_randcode": {"random_codes": True}, "qfl_pauli": {"pauli_readout": True},
}
NAME = re.compile(r"^(qfl_fedavg|qfl_fedprox_full|qfl_fedprox|qfl_weighted|qfl_logitadj|qfl_fco|qfl_frozen|qfl_randcode|qfl_pauli)"
                  r"_(iid|dirichlet)_a([\d.]+)(?:_mu[\d.e-]+)?(?:_lam[\d.]+)?(?:_tau[\d.]+)?_c(\d+)_e\d+_lr[\d.e-]+_s(\d+)\.pt$")


def run(dataset="fashionmnist", folder=None, shots_list=(10, 100, 1000, 10000), seed=0, repeats=1):
    if folder is None:
        folder = "saved" if dataset == "fashionmnist" else dataset
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
        exact = evaluate(model, test_loader)["accuracy"]
        # the shots are drawn again with a new sampling seed in every repeat
        for sample_seed in range(seed, seed + repeats):
            noisy = shot_accuracies(model, test_loader, list(shots_list), pairs=(kind == "qfl_pauli"), seed=sample_seed)
            for shots, acc in noisy.items():
                rows.append({"kind": kind, "partition": partition, "alpha": float(alpha), "clients": int(clients),
                             "seed": int(run_seed), "sample_seed": sample_seed, "shots": shots, "accuracy": acc,
                             "exact_accuracy": exact})
            print(f"{fname[:-3]} (sampling seed {sample_seed}): exact {exact:.4f} | "
                  + " | ".join(f"{s} shots {a:.4f}" for s, a in noisy.items()) + f" | {time.time() - start:.0f}s", flush=True)

    out = os.path.join("results", "tables", folder, "shots.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashionmnist", choices=list(DATASETS))
    parser.add_argument("--folder",  type=str, default=None)
    parser.add_argument("--shots",   type=int, nargs="+", default=[10, 100, 1000, 10000])
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    run(dataset=args.dataset, folder=args.folder, shots_list=args.shots, seed=args.seed, repeats=args.repeats)
