# Hybrid Quantum-Classical Federated Learning under Non-IID Client Data Distributions

## Overview

This repository presents an empirical study of non-IID data heterogeneity in Hybrid Quantum-Classical Federated Learning (QFL). We investigate how skewed client data distributions degrade model performance when local models are Hybrid Quantum Neural Networks (Hybrid QNNs), and evaluate two mitigation strategies that have not been previously assessed in the QFL setting.

The research gap addressed: Zhao et al. (2023) proved theoretically that non-IID client distributions cause weight divergence in QFL. No prior work has empirically benchmarked mitigation strategies (proximal regularization, distribution-aware aggregation) under controlled heterogeneity in a hybrid QFL system. This work fills that gap.

---

## Research Contributions

1. **Empirical benchmark** of non-IID degradation in Hybrid QFL across three data heterogeneity conditions (IID, Dirichlet label-skew at α=0.5 and α=0.1) and three client scales (10, 20, 50).

2. **Quantum FedProx** — a novel adaptation of FedProx (Li et al., 2020) that applies the proximal drift penalty specifically to variational quantum circuit (VQC) parameters, preventing quantum gate angles from diverging under heterogeneous client distributions.

3. **Distribution-aware weighted aggregation** — server-side aggregation that weights client contributions by both data volume and Shannon entropy of each client's label distribution, reducing the influence of severely skewed clients.

4. **Systematic EMD analysis** — application of the Earth Mover's Distance proxy (Zhao et al., 2023) as an experimental metric to quantify non-IID level across all partition configurations.

---

## System Architecture

**Classical feature extraction → Quantum encoding → VQC → Classification**

Each client's local model is a Hybrid QNN:
- **Feature extractor:** 2-layer CNN (Conv2d 1→8→16, MaxPool, compressing 28×28 image to a 4-dimensional latent vector)
- **Quantum encoding:** Angle embedding — each of the 4 features maps to a qubit rotation angle via `tanh(x) × π`
- **Variational Quantum Circuit:** 4 qubits, 2 StronglyEntanglingLayers (24 trainable quantum parameters), Pauli-Z expectation measurement
- **Classification head:** Linear(4 → 10) over PauliZ expectation values

**Federation strategies compared:**

| Strategy | Description |
|---|---|
| FedAvg (Classical) | Standard weighted averaging on classical CNN |
| QFL FedAvg | FedAvg applied to Hybrid QNN parameters |
| Quantum FedProx | FedAvg + proximal penalty on VQC parameters during local training |
| Weighted Aggregation | Aggregation weighted by data volume × label distribution entropy |

---

## Baseline Results (Classical FedAvg, 10 clients, 3 rounds, Fashion-MNIST)

| Data Partition | Heterogeneity | EMD Proxy | Accuracy |
|---|---|---|---|
| IID | Control | 0.037 | **84.57%** |
| Dirichlet α=0.5 | Moderate | 1.18 | **83.48%** |
| Dirichlet α=0.1 | Severe | 1.71 | **72.22%** |

Non-IID accuracy degradation (IID → α=0.1): **−12.35 percentage points**

---

## Repository Structure

```
hybrid-qfl-noniid/
├── data/
│   ├── utils.py                  — Dataset loading and partition visualization
│   ├── iid_partition.py          — Uniform IID split (control condition)
│   ├── label_skew.py             — Dirichlet label-skew partition + EMD proxy metric
│   └── quantity_skew.py          — Log-normal quantity-skew partition
├── models/
│   ├── classical_cnn.py          — Classical CNN baseline (105,866 parameters)
│   ├── vqc_ansatz.py             — PennyLane VQC: AngleEmbedding + StronglyEntanglingLayers
│   └── hybrid_qnn.py             — Hybrid QNN: CNN extractor + VQC + classification head (26,574 params)
├── federation/
│   ├── fedavg.py                 — FedAvg: local training + weighted parameter aggregation
│   ├── quantum_fedprox.py        — Quantum FedProx: proximal penalty on VQC parameters
│   └── weighted_aggregation.py   — Entropy-weighted server-side aggregation
├── evaluation/
│   └── classical_metrics.py      — Test loss and accuracy evaluation
├── experiments/
│   ├── run_classical_baseline.py — Classical FL baseline experiment runner
│   ├── run_qfl_fedavg.py         — QFL with standard FedAvg
│   ├── run_fedprox.py            — QFL with Quantum FedProx
│   └── run_weighted_agg.py       — QFL with distribution-aware weighted aggregation
├── notebooks/
│   ├── plot_baseline.ipynb       — Convergence curve and accuracy bar chart generation
│   └── plot_results.ipynb        — Results tables and paper figures from all runs
├── results/tables/               — CSV output from all experiment runs
├── Lit review/                   — 6 reference papers (PDF)
├── verify_setup.py               — End-to-end environment verification script
├── requirements.txt
└── environment.yml
```

---

## Setup

```bash
pip install -r requirements.txt
```

Verify environment and data pipeline:
```bash
python verify_setup.py
```

---

## Running Experiments

```bash
# Classical FL baseline
python experiments/run_classical_baseline.py --partition iid --rounds 5
python experiments/run_classical_baseline.py --partition dirichlet --alpha 0.5 --rounds 5
python experiments/run_classical_baseline.py --partition dirichlet --alpha 0.1 --rounds 5

# QFL with FedAvg
python experiments/run_qfl_fedavg.py --partition dirichlet --alpha 0.1 --rounds 5

# Quantum FedProx
python experiments/run_fedprox.py --partition dirichlet --alpha 0.1 --rounds 5 --mu 0.01

# Weighted aggregation
python experiments/run_weighted_agg.py --partition dirichlet --alpha 0.1 --rounds 5 --lam 0.5
```

Results are saved to `results/tables/` as CSV files.

---

## Key References

- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.
- Li et al. (2020). Federated Optimization in Heterogeneous Networks (FedProx). *MLSys*.
- Huang et al. (2022). Quantum Federated Learning with Decentralized Data. *Quantum Machine Intelligence*.
- Zhao et al. (2023). Non-IID Quantum Federated Learning with One-Shot Communication Complexity. *Quantum Machine Intelligence*.
- Chehimi & Saad (2022). Quantum Federated Learning with Quantum Data. *Quantum Machine Intelligence*.

---

## Team

- Arya Mulay
- Nakul Thombare
- Keshav Sukhija

**Faculty Supervisor:** Dr. Aswani Kumar Cherukuri, Professor & Director, C-FAIR, VIT Vellore

**Frameworks:** PennyLane 0.45 · PyTorch 2.14 · Flower 1.36
