# Hybrid Quantum-Classical Federated Learning under Non-IID Client Data Distributions

## Overview

In federated learning (FL) many clients train one shared model without sharing their data: each client trains on its own data and a server averages the models. In practice the clients hold very different data (non-IID), for example hospitals that see different diseases.

This repository studies FL when every client's model is a **hybrid quantum neural network**: a small CNN turns each image into 4 angles, a 4-qubit variational quantum circuit (VQC) processes them, and a readout layer turns the 4 measured Pauli-Z values into class scores. We create label skew with a Dirichlet split, measure it with an EMD proxy, and compare methods on the same seeds with paired tests.

What we found:

- The hybrid model is much more fragile under label skew than a classical CNN (Fashion-MNIST: it loses 41.4 accuracy points from IID to Dirichlet α = 0.1, the CNN loses 7.9).
- The standard fixes we tested (a proximal term on the quantum parameters or on all parameters, entropy-weighted aggregation, a calibrated local loss) give no reliable gain over 8 seeds.
- The cause is **readout collapse**: the readout maps only 4 measured values to 10 classes, and averaging it across skewed clients leaves some classes with no region where they can win, so they can never be predicted. At α = 0.1 the readout learned by FedAvg leaves on average 1.7 (Fashion-MNIST), 2.4 (MNIST) and 1.8 (PathMNIST) classes unreachable, up to 4 on one seed.
- **Fixed class observables (FCO)** freeze the readout as one fixed quantum observable per class, the same on every client and never averaged. Every class stays reachable by construction.

## Method: fixed class observables (FCO)

The readout weights are fixed to `W = s · V` with zero bias and are never trained, so the score of class c is a fixed observable measured on the circuit state:

```
f_c(x) = <psi(x)| O_c |psi(x)>,   O_c = s * sum_i V_ci Z_i
```

The 10 rows of `V` are the 5 corners of a regular simplex in 4 dimensions and their 5 opposites (unit length, every pair at least 75.5 degrees apart), and `s = 5`. All clients train only the CNN encoder and the circuit towards the same class targets; FedAvg averages those parts only. Code: `class_codes` and the `fixed_readout` option in `models/hybrid_qnn.py`.

Variants and controls used to test it:

| Name | What it is |
|---|---|
| Pauli observables | Class c is read from its own Pauli-Z string: 4 single qubits and the 6 two-qubit parities Z_i Z_j, all from the same measurement shots |
| Random frozen head | The random initial readout, frozen (the FedBABU approach) |
| Random unit codes | Random directions of the same length as FCO's, no bias, frozen |
| Classical twin | The same model with the circuit replaced by a classical 4 x 4 layer |
| Twin + FCO | The twin with the FCO readout |

## Main results

Test accuracy (%), mean of the last 5 of 20 rounds, mean over seeds; 10 clients, 1 local epoch, batch size 32, Adam (lr 0.003 hybrid, 0.001 CNN).

| Dataset | Split | QFL FedAvg | FCO (ours) | Classical twin | Classical CNN |
|---|---|---|---|---|---|
| Fashion-MNIST | IID | 85.01 | **87.83** | 88.57 | 90.14 |
| Fashion-MNIST | Dirichlet α = 0.5 | 75.81 | **85.18** | 85.51 | 88.93 |
| Fashion-MNIST | Dirichlet α = 0.1 | 43.61 | **60.37** | 69.01 | 82.24 |
| MNIST | IID | 97.68 | 97.83 | 98.68 | 99.05 |
| MNIST | Dirichlet α = 0.5 | 88.13 | **97.37** | 98.40 | 98.93 |
| MNIST | Dirichlet α = 0.1 | 61.58 | **81.20** | 83.52 | 90.25 |
| PathMNIST | IID | 77.40 | 76.39 | 79.48 | 81.88 |
| PathMNIST | Dirichlet α = 0.5 | 67.59 | **72.22** | 76.25 | 79.55 |
| PathMNIST | Dirichlet α = 0.1 | 38.36 | 37.84 | 50.19 | 61.72 |

Seeds: Fashion-MNIST 8 for FedAvg, FCO and the twin (the CNN 3 at IID and α = 0.5); MNIST and PathMNIST 5 at α = 0.1 and α = 0.5 and 3 at IID (the CNN 5 at IID). Same-seed gains of FCO over FedAvg: Fashion-MNIST +2.82 / +9.37 / +16.77 at IID / α = 0.5 / α = 0.1 (p = 0.012 / 0.004 / 0.0004); MNIST +9.24 at α = 0.5 (5 of 5 seeds, p = 0.048) and +19.62 at α = 0.1 (5 of 5, p = 0.006); PathMNIST +4.63 at α = 0.5 (5 of 5, p = 0.020).

More clients (Fashion-MNIST, α = 0.1, 5 seeds): the FCO gain over FedAvg is +15.5 with 10 clients, +16.4 with 20 and +25.0 with 50.

Readout ablations (Fashion-MNIST, α = 0.1, 8 seeds): random frozen head 28.33, random unit codes 54.97, Pauli observables 57.23, FCO 60.37.

Finite measurement shots (saved models): with 1,000 shots per reading FCO stays within 0.4 points of its noiseless accuracy on every dataset and split (every model within 0.8), and with 100 shots FCO keeps its lead over FedAvg at α = 0.1 (Fashion-MNIST +16.9, MNIST +16.7). With only 10 shots FCO degrades more than FedAvg.

Quantum state analysis and hardware noise (saved models): the circuit trained with FedAvg ends in states close to a few bitstrings, while the FCO circuit stays entangled and spread over many outcomes. On every dataset and split, FCO has higher Meyer-Wallach entanglement (+0.19 to +0.34) and higher measurement entropy (+1.05 to +1.94 bits) than FedAvg on every seed. On a density-matrix simulator, FCO's predictions do not change at all with readout errors (tested up to 10% flipped bits) and change by at most 0.7 points with a depolarizing error of 5% after every gate, while FedAvg loses 1.4 to 13.3 points. With that gate noise FCO beats FedAvg on every seed of every split, including PathMNIST at α = 0.1 (+10.0, 5 of 5 seeds).

What it does not do: on PathMNIST at α = 0.1 FCO removes the readout collapse and fewer classes are lost, but accuracy does not improve, because the image features themselves degrade (even the classical CNN loses 20 points there). The classical twin and the classical CNN remain more accurate than the hybrid model with FCO.

## Repository structure

```
hybrid-qfl-noniid/
├── data/
│   ├── utils.py                  - load_dataset (Fashion-MNIST, MNIST, PathMNIST), DATASETS table, partition plot
│   ├── iid_partition.py          - equal random split (control)
│   ├── label_skew.py             - Dirichlet label-skew split and the EMD proxy
│   └── quantity_skew.py          - log-normal quantity-skew split
├── models/
│   ├── classical_cnn.py          - classical CNN reference (105,866 parameters)
│   ├── vqc_ansatz.py             - 4-qubit PennyLane circuit: angle embedding + 2 strongly entangling layers
│   └── hybrid_qnn.py             - hybrid model (26,574 parameters) with the FCO, Pauli, frozen, random-code and twin options
├── federation/
│   ├── fedavg.py                 - local training and size-weighted averaging
│   ├── quantum_fedprox.py        - proximal term on the quantum (or all) parameters
│   ├── weighted_aggregation.py   - size and label-entropy weighted averaging
│   └── logit_adjust.py           - calibrated local loss
├── evaluation/
│   ├── classical_metrics.py      - test loss, accuracy and per-class accuracy
│   ├── client_drift.py           - per-layer drift and agreement of the client updates
│   ├── shot_noise.py             - test accuracy with a finite number of measurement shots
│   ├── entanglement.py           - entanglement, measurement entropy and pair correlations of the circuit states
│   └── gate_noise.py             - test accuracy with gate noise and readout errors (density-matrix simulator)
├── experiments/
│   ├── run_qfl_fedavg.py         - hybrid model with FedAvg (and every readout variant)
│   ├── run_fedprox.py            - hybrid model with FedProx
│   ├── run_weighted_agg.py       - hybrid model with weighted aggregation
│   ├── run_logit_adjust.py       - hybrid model with the calibrated loss
│   ├── run_classical_baseline.py - classical CNN with FedAvg
│   ├── run_grid.py               - runs many settings and seeds in parallel, skips finished runs
│   ├── eval_shots.py             - finite-shot evaluation of the saved models
│   ├── eval_entanglement.py      - entanglement of the saved models, at the start and after training
│   └── eval_noise.py             - gate noise and readout error test of the saved models
├── notebooks/
│   ├── plot_baseline_{fashionmnist,mnist,pathmnist}.ipynb - data exploration, client splits, EMD, classical baseline
│   ├── plot_results_{fashionmnist,mnist,pathmnist}.ipynb  - results tables, same-seed gains with p-values, figures
│   └── plot_diagnostics.ipynb    - reproducibility, 50 rounds, lost classes, readout collapse, clients, shots, entanglement, noise
├── results/                      - tables, models, logs and plots (created by the runs, not in git)
├── Lit review/                   - reference papers (PDF)
├── verify_setup.py               - checks the environment and the data pipeline
├── requirements.txt
└── environment.yml
```

## Setup

```bash
pip install -r requirements.txt
python verify_setup.py
```

Tested with Python 3.14, PyTorch 2.14 (CPU), torchvision 0.29 and PennyLane 0.45.1. PathMNIST needs the `medmnist` package; `pip install medmnist --no-deps` is enough. Everything runs on a normal CPU.

## Running experiments

Run everything from the repository folder. One run:

```bash
python experiments/run_qfl_fedavg.py --partition dirichlet --alpha 0.1 --rounds 20 --lr 0.003 --seed 42 --fixed_readout
```

Readout options of `run_qfl_fedavg.py`: `--fixed_readout` (FCO), `--pauli_readout`, `--frozen_readout`, `--random_codes`, `--twin` (and `--twin --fixed_readout`). Every runner takes `--dataset {fashionmnist,mnist,pathmnist}`, `--clients`, `--rounds`, `--epochs`, `--batch_size`, `--lr`, `--partition {iid,dirichlet,quantity}`, `--alpha`, `--seed` and `--folder`.

Many runs at once (the setting of the paper; finished runs are skipped):

```bash
python experiments/run_grid.py --strategies fedavg fco twin --splits iid dir0.5 dir0.1 --seeds 42 43 44 45 46 47 48 49 --workers 3
python experiments/run_grid.py --strategies classical --lr 0.001 --splits iid dir0.5 dir0.1 --seeds 42 43 44 --workers 3
python experiments/run_grid.py --dataset mnist --strategies fedavg fco fedprox_full twin twin_fco --splits dir0.1 --seeds 42 43 44 45 46 --workers 3
python experiments/run_grid.py --strategies fedavg fco --splits dir0.1 --clients 50 --seeds 42 43 44 45 46 --folder clients --workers 3
```

Grid strategies: `fedavg`, `fedprox`, `fedprox_full`, `weighted`, `logit_adjust`, `classical`, `fco`, `pauli`, `frozen`, `randcode`, `twin`, `twin_fco`. Splits: `iid` or `dirX` for Dirichlet α = X.

Finite-shot evaluation of the saved models:

```bash
python experiments/eval_shots.py --dataset fashionmnist
```

Entanglement of the circuit states (Meyer-Wallach measure, measurement entropy, pair correlations) and test accuracy under gate noise and readout errors on PennyLane's density-matrix simulator `default.mixed`:

```bash
python experiments/eval_entanglement.py --dataset fashionmnist
python experiments/eval_noise.py --dataset fashionmnist
```

Every run writes a CSV to `results/tables/<folder>/` (test accuracy, loss, per-class accuracy, per-layer drift and agreement for every round, and the measured EMD) and its final model to `results/models/<folder>/`. Runs on MNIST and PathMNIST go to their own folders. The same seed gives the same client split, the same starting weights and the same data order, so every result can be repeated exactly. The notebooks read these folders and produce all tables and figures.

## References

- McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS.
- Li, T., et al. (2020). Federated optimization in heterogeneous networks (FedProx). MLSys.
- Hsu, T.-M. H., Qi, H., & Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. arXiv:1909.06335.
- Huang, R., Tan, X., & Xu, Q. (2022). Quantum federated learning with decentralized data. IEEE Journal of Selected Topics in Quantum Electronics.
- Zhao, H. (2023). Non-IID quantum federated learning with one-shot communication complexity. Quantum Machine Intelligence.
- Chehimi, M., & Saad, W. (2022). Quantum federated learning with quantum data. ICASSP.
- Oh, J., Kim, S., & Yun, S.-Y. (2022). FedBABU: Toward enhanced representation for federated image classification. ICLR.
- Li, Z., et al. (2023). No fear of classifier biases: Neural collapse inspired federated learning with synthetic and fixed classifier. ICCV.
- Zhang, J., et al. (2022). Federated learning with label distribution skew via logits calibration. ICML.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST. arXiv:1708.07747.
- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.
- Yang, J., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data.

## Team

- Arya Mulay
- Nakul Thombare
- Keshav Sukhija

**Faculty supervisor:** Dr. Aswani Kumar Cherukuri, Professor and Director, C-FAIR, VIT Vellore
