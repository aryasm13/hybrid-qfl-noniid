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
from evaluation import evaluate, state_properties
from eval_shots import KINDS, NAME


def run(dataset="fashionmnist", folder=None):
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
        for stage in ["init", "trained"]:
            # init: the same starting weights as the run (the runners seed torch right before building the model)
            torch.manual_seed(int(run_seed))
            model = HybridQNN(in_channels=in_channels, num_classes=num_classes, **KINDS[kind])
            if stage == "trained":
                model.load_state_dict(torch.load(os.path.join(model_dir, fname)))
            props = state_properties(model, test_loader)
            rows.append({"kind": kind, "partition": partition, "alpha": float(alpha), "clients": int(clients),
                         "seed": int(run_seed), "stage": stage, **props,
                         "accuracy": evaluate(model, test_loader)["accuracy"]})
        print(f"{fname[:-3]}: entanglement {rows[-2]['meyer_wallach']:.3f} -> {rows[-1]['meyer_wallach']:.3f} | "
              f"z_norm {rows[-1]['z_norm']:.3f} | zz_connected {rows[-1]['zz_connected']:.3f} | "
              f"entropy {rows[-1]['entropy']:.2f} | acc {rows[-1]['accuracy']:.4f} | {time.time() - start:.0f}s", flush=True)

    out = os.path.join("results", "tables", folder, "entanglement.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashionmnist", choices=list(DATASETS))
    parser.add_argument("--folder",  type=str, default=None)
    args = parser.parse_args()

    run(dataset=args.dataset, folder=args.folder)
