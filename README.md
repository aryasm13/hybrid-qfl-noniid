# Hybrid Quantum-Classical Federated Learning under Non-IID Client Data

An empirical study comparing Federated Learning strategies on heterogeneous (non-IID) client data using a Hybrid Quantum-Classical Neural Network.

**Review 1 Status: Classical FL baseline running. Data pipeline complete. Hybrid QNN next.**

---

## The Research Problem

Federated Learning (FL) lets multiple clients train a shared model without sharing raw data — privacy preserved. But it assumes client data is uniform (IID). In reality, clients have very different data distributions. This is the **non-IID problem** and it causes accuracy to drop sharply.

This problem is well-studied in classical FL. In **Quantum Federated Learning (QFL)**, it has only been proven theoretically (Zhao et al., 2023) but never empirically benchmarked with mitigation strategies. That is our gap.

We implement a **Hybrid QNN** — classical CNN feature extractor feeding a Variational Quantum Circuit (VQC) — inside a federated setup, and compare three strategies to handle non-IID data.

---

## What's Implemented

### Data Pipeline
| File | What it does |
|---|---|
| `data/iid_partition.py` | Splits data uniformly across clients — control condition |
| `data/label_skew.py` | Dirichlet(α) partition — lower α = more skewed client data |
| `data/quantity_skew.py` | Log-normal volume split — unequal sample counts per client |
| `data/utils.py` | Fashion-MNIST loader, label extraction, stacked-bar visualization |

EMD proxy (Earth Mover's Distance) computed per partition to quantify non-IID level.

### Classical Baseline (done)
| File | What it does |
|---|---|
| `models/classical_cnn.py` | 2-layer CNN — 105,866 parameters, benchmarks against quantum model |
| `federation/fedavg.py` | FedAvg — local training + weighted global aggregation |
| `evaluation/classical_metrics.py` | Test loss + accuracy per round |
| `experiments/run_classical_baseline.py` | End-to-end runner, logs CSV to `results/tables/` |

### Baseline Results (3 rounds, 10 clients, Fashion-MNIST)

| Partition | α | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| IID | — | 74.69% | 81.31% | **84.57%** |
| Dirichlet label-skew | 0.5 | 71.59% | 80.93% | **83.48%** |
| Dirichlet label-skew | 0.1 | 44.27% | 58.58% | **72.22%** |

The 12% accuracy gap between IID and α=0.1 is the non-IID degradation we are solving.

---

## What's Coming Next

### Hybrid QNN (in progress)
- `models/vqc_ansatz.py` — PennyLane VQC: angle embedding + strongly entangling layers
- `models/hybrid_qnn.py` — CNN feature extractor → VQC → classification head

### Mitigation Strategies
- `federation/weighted_aggregation.py` — weight clients by data size + class balance entropy
- `federation/quantum_fedprox.py` — proximal penalty on VQC parameter drift

### Full Experiment Grid
```
Strategies:   [Classical FedAvg, QFL FedAvg, Weighted Agg, Quantum FedProx]
Partitions:   [IID, Dirichlet α=0.5, Dirichlet α=0.1, Quantity-skew]
Clients:      [10, 20, 50]
Rounds:       [50]
```

---

## Directory Structure

```
hybrid-qfl-noniid/
├── data/                    ← done: all 3 partitioners + Fashion-MNIST loader
│   ├── utils.py
│   ├── iid_partition.py
│   ├── label_skew.py
│   └── quantity_skew.py
├── models/                  ← done: classical CNN | next: VQC + HybridQNN
│   └── classical_cnn.py
├── federation/              ← done: FedAvg | next: weighted agg + quantum FedProx
│   └── fedavg.py
├── evaluation/              ← done: loss + accuracy | next: macro F1 + circuit metrics
│   └── classical_metrics.py
├── experiments/             ← done: classical baseline runner
│   └── run_classical_baseline.py
├── results/tables/          ← CSV results from all runs
├── Lit review/              ← 6 reference papers
└── verify_setup.py          ← end-to-end Week 1 check (all passing)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Verify everything works:
```bash
python verify_setup.py
```

---

## Running the Classical Baseline

```bash
# IID (control)
python experiments/run_classical_baseline.py --partition iid --rounds 5

# Non-IID moderate skew
python experiments/run_classical_baseline.py --partition dirichlet --alpha 0.5 --rounds 5

# Non-IID high skew
python experiments/run_classical_baseline.py --partition dirichlet --alpha 0.1 --rounds 5

# Quantity skew
python experiments/run_classical_baseline.py --partition quantity --rounds 5
```

---

## Commit History

```
70f9103  feat(experiments): classical FL baseline runner with IID and non-IID results
c783184  feat(federation): FedAvg aggregation and evaluation metrics
4395d73  feat(models): add classical CNN baseline
d175ed6  updated readme, added module placeholders and verify script
c788bcd  feat(data): implement iid, dirichlet label-skew, and quantity-skew partitioners
b681b09  feat(env): add dependencies and environment configuration
```

---

## Team

- **Arya Mulay** — Quantum FedProx, evaluation metrics, experiment runner
- **Nakul Thombare** — Data pipeline, Hybrid QNN architecture, FedAvg
- **Keshav Sukhija** — Huang 2022 reproduction, weighted aggregation, paper writing
- **Faculty:** Dr. Aswani Kumar Cherukuri (C-FAIR, VIT)
- **Reviewer:** Prasenjit Roy (Asst. Prof, School of Advanced Sciences, VIT)
- **Frameworks:** PennyLane 0.45, PyTorch 2.14, Flower 1.36
