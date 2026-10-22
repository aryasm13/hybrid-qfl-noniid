# Hybrid Quantum-Classical Federated Learning under Non-IID Client Data Distributions

## Overview

In federated learning (FL) many clients train one shared model without sharing their data: each client trains on its own data and a server averages the models. In practice the clients hold very different data (non-IID), for example hospitals that see different diseases.

This repository studies FL when every client's model is a **hybrid quantum neural network**: a small CNN turns each image into 4 angles, a 4-qubit variational quantum circuit (VQC) processes them, and a readout layer turns the 4 measured Pauli-Z values into class scores. We create label skew with a Dirichlet split, measure it with an EMD proxy, and compare methods on the same seeds with paired tests.

What we found:

- The hybrid model is much more fragile under label skew than a classical CNN (Fashion-MNIST: it loses 41.4 accuracy points from IID to Dirichlet α = 0.1, the CNN loses 7.9).
- The standard fixes we tested (a proximal term on the quantum parameters or on all parameters, entropy-weighted aggregation, a calibrated local loss) give no reliable gain over 8 seeds.
- The cause is **readout collapse**: the readout maps only 4 measured values to 10 classes, and averaging it across skewed clients leaves some classes with no region where they can win, so they can never be predicted. At α = 0.1 the readout learned by FedAvg leaves on average 1.7 (Fashion-MNIST), 2.4 (MNIST), 1.8 (PathMNIST) and 3.0 (CIFAR-10) classes unreachable, up to 6 on one seed.
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

| Dataset | Split | QFL FedAvg | FCO (proposed method) | Classical twin | Classical CNN |
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
| CIFAR-10 | IID | 46.84 | **50.78** | 55.05 | 63.13 |
| CIFAR-10 | Dirichlet α = 0.5 | 38.96 | **45.76** | 50.37 | 60.88 |
| CIFAR-10 | Dirichlet α = 0.1 | 22.82 | **29.83** | 32.24 | 52.97 |

Seeds: Fashion-MNIST 8 for FedAvg, FCO and the twin (the CNN 3 at IID and α = 0.5); MNIST, PathMNIST and CIFAR-10 5 at α = 0.1 and α = 0.5 and 3 at IID (the CNN 5 at IID). Same-seed gains of FCO over FedAvg: Fashion-MNIST +2.82 / +9.37 / +16.77 at IID / α = 0.5 / α = 0.1 (p = 0.012 / 0.004 / 0.0004); MNIST +9.24 at α = 0.5 (5 of 5 seeds, p = 0.048) and +19.62 at α = 0.1 (5 of 5, p = 0.006); PathMNIST +4.63 at α = 0.5 (5 of 5, p = 0.020); CIFAR-10 +3.94 / +6.79 / +7.01 at IID / α = 0.5 / α = 0.1 (3 of 3, 4 of 5 and 5 of 5 seeds, p = 0.047 / 0.031 / 0.017).

CIFAR-10 (colour photos, resized from 32 x 32 to 28 x 28 so that every model is unchanged) is hard for every model at this size: even the classical CNN reaches only 63% with IID data, so its results are best read relative to the CNN. From IID to α = 0.1 the hybrid model with FedAvg keeps 49% of its accuracy, with FCO 59%, and the CNN 84%.

More clients (Fashion-MNIST, α = 0.1, 5 seeds): the FCO gain over FedAvg is +15.5 with 10 clients, +16.4 with 20, +25.0 with 50 and +15.2 with 100, better on every seed each time (p = 0.009, 0.033, 0.001, 0.011). FCO stays between 58% and 64% from 10 to 100 clients, FedAvg between 39% and 47%.

Readout ablations (Fashion-MNIST, α = 0.1, 8 seeds): random frozen head 28.33, random unit codes 54.97, Pauli observables 57.23, FCO 60.37.

Finite measurement shots (saved models, each tested 5 times with a new sampling seed; the standard deviation between the repeats is at most 0.5 points): with 1,000 shots per reading FCO stays within 0.5 points of its noiseless accuracy on every dataset and split (every model within 0.8), and with 100 shots FCO keeps its lead over FedAvg at α = 0.1 on Fashion-MNIST (+17.1), MNIST (+16.9) and CIFAR-10 (+5.3). With only 10 shots FCO degrades more than FedAvg.

Quantum state analysis and hardware noise (saved models): the circuit trained with FedAvg ends in states close to a few bitstrings, while the FCO circuit stays entangled and spread over many outcomes. On every dataset and split, FCO has higher Meyer-Wallach entanglement (+0.19 to +0.34) and higher measurement entropy (+1.05 to +1.94 bits) than FedAvg on every seed. On a density-matrix simulator, FCO's predictions do not change at all with readout errors (all 42 FCO models, tested up to 10% flipped bits) and change by at most 0.7 points on average with a depolarizing error of 5% after every gate, while FedAvg loses 1.4 to 13.3 points. With that gate noise FCO beats FedAvg on every seed of every split, including PathMNIST at α = 0.1 (+10.0, 5 of 5 seeds).

What it does not do: on PathMNIST at α = 0.1 FCO removes the readout collapse and fewer classes are lost, but accuracy does not improve, because the image features themselves degrade (even the classical CNN loses 20 points there). On CIFAR-10 at α = 0.1 FCO helps the classical twin even more than the hybrid model (+10.0 against +7.0), so the result that FCO helps the quantum model more than its classical twin holds on Fashion-MNIST and MNIST but not on CIFAR-10. The classical twin and the classical CNN remain more accurate than the hybrid model with FCO.

## Repository structure

```
hybrid-qfl-noniid/
├── data/
│   ├── utils.py                  - load_dataset (Fashion-MNIST, MNIST, PathMNIST, CIFAR-10), DATASETS table, partition plot
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
│   ├── plot_baseline_{fashionmnist,mnist,pathmnist,cifar10}.ipynb - data exploration, client splits, EMD, classical baseline
│   ├── plot_results_{fashionmnist,mnist,pathmnist,cifar10}.ipynb  - results tables, same-seed gains with p-values, figures
│   └── plot_diagnostics.ipynb    - reproducibility, 50 rounds, lost classes, readout collapse, clients, shots, entanglement, noise
├── results/                      - tables, models, logs and plots (created by the runs, not in git)
├── Lit review/                   - the reference papers (PDF), numbered as in the report
├── verify_setup.py               - checks the environment and the data pipeline
├── requirements.txt
└── environment.yml
```

## Setup

```bash
pip install -r requirements.txt
python verify_setup.py
```

Tested with Python 3.14, PyTorch 2.14 (CPU), torchvision 0.29 and PennyLane 0.45.1. PathMNIST needs the `medmnist` package; `pip install medmnist --no-deps` is enough. Fashion-MNIST, MNIST and CIFAR-10 download through torchvision. Everything runs on a normal CPU.

## Running experiments

Run everything from the repository folder. One run:

```bash
python experiments/run_qfl_fedavg.py --partition dirichlet --alpha 0.1 --rounds 20 --lr 0.003 --seed 42 --fixed_readout
```

Readout options of `run_qfl_fedavg.py`: `--fixed_readout` (FCO), `--pauli_readout`, `--frozen_readout`, `--random_codes`, `--twin` (and `--twin --fixed_readout`). Every runner takes `--dataset {fashionmnist,mnist,pathmnist,cifar10}`, `--clients`, `--rounds`, `--epochs`, `--batch_size`, `--lr`, `--partition {iid,dirichlet,quantity}`, `--alpha`, `--seed` and `--folder`.

Many runs at once (the setting of the paper; finished runs are skipped):

```bash
python experiments/run_grid.py --strategies fedavg fco twin --splits iid dir0.5 dir0.1 --seeds 42 43 44 45 46 47 48 49 --workers 3
python experiments/run_grid.py --strategies classical --lr 0.001 --splits iid dir0.5 dir0.1 --seeds 42 43 44 --workers 3
python experiments/run_grid.py --dataset mnist --strategies fedavg fco fedprox_full twin twin_fco --splits dir0.1 --seeds 42 43 44 45 46 --workers 3
python experiments/run_grid.py --dataset cifar10 --strategies fedavg fco twin twin_fco --splits dir0.5 dir0.1 --seeds 42 43 44 45 46 --workers 3
python experiments/run_grid.py --strategies fedavg fco --splits dir0.1 --clients 50 --seeds 42 43 44 45 46 --folder clients --workers 3
python experiments/run_grid.py --strategies fedavg fco --splits dir0.1 --clients 100 --seeds 42 43 44 45 46 --folder clients --workers 3
```

Grid strategies: `fedavg`, `fedprox`, `fedprox_full`, `weighted`, `logit_adjust`, `classical`, `fco`, `pauli`, `frozen`, `randcode`, `twin`, `twin_fco`. Splits: `iid` or `dirX` for Dirichlet α = X.

Finite-shot evaluation of the saved models (`--repeats 5` draws the shots 5 times with different sampling seeds):

```bash
python experiments/eval_shots.py --dataset fashionmnist --repeats 5
```

Entanglement of the circuit states (Meyer-Wallach measure, measurement entropy, pair correlations) and test accuracy under gate noise and readout errors on PennyLane's density-matrix simulator `default.mixed`:

```bash
python experiments/eval_entanglement.py --dataset fashionmnist
python experiments/eval_noise.py --dataset fashionmnist
```

Every run writes a CSV to `results/tables/<folder>/` (test accuracy, loss, per-class accuracy, per-layer drift and agreement for every round, and the measured EMD) and its final model to `results/models/<folder>/`. Runs on MNIST, PathMNIST and CIFAR-10 go to their own folders. The same seed gives the same client split, the same starting weights and the same data order, so every result can be repeated exactly. The notebooks read these folders and produce all tables and figures.

## References

The 31 references of the project report, numbered as in the report (IEEE style). The papers are in `Lit review/`, with file names that start with the same number (arXiv or open-access versions where the published version is not open). Reference 24 is a textbook and is not included.

1. H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS), PMLR, vol. 54, 2017, pp. 1273-1282.
2. T.-M. H. Hsu, H. Qi, and M. Brown, "Measuring the effects of non-identical data distribution for federated visual classification," arXiv:1909.06335, 2019.
3. Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, "Federated learning with non-IID data," arXiv:1806.00582, 2018.
4. T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks," in Proc. Mach. Learn. Syst. (MLSys), vol. 2, 2020, pp. 429-450.
5. J. Zhang, Z. Li, B. Li, J. Xu, S. Wu, S. Ding, and C. Wu, "Federated learning with label distribution skew via logits calibration," in Proc. 39th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 162, 2022, pp. 26311-26329.
6. X.-C. Li and D.-C. Zhan, "FedRS: Federated learning with restricted softmax for label distribution non-IID data," in Proc. 27th ACM SIGKDD Conf. Knowl. Discov. Data Min. (KDD), 2021.
7. J. Oh, S. Kim, and S.-Y. Yun, "FedBABU: Toward enhanced representation for federated image classification," in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.
8. Z. Li, X. Shang, R. He, T. Lin, and C. Wu, "No fear of classifier biases: Neural collapse inspired federated learning with synthetic and fixed classifier," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 5319-5329.
9. J. Preskill, "Quantum computing in the NISQ era and beyond," Quantum, vol. 2, p. 79, 2018, doi: 10.22331/q-2018-08-06-79.
10. M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, "Variational quantum algorithms," Nat. Rev. Phys., vol. 3, no. 9, pp. 625-644, 2021, doi: 10.1038/s42254-021-00348-9.
11. M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, "Circuit-centric quantum classifiers," Phys. Rev. A, vol. 101, no. 3, Art. no. 032308, 2020.
12. M. Chehimi and W. Saad, "Quantum federated learning with quantum data," in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2022, pp. 8617-8621.
13. R. Huang, X. Tan, and Q. Xu, "Quantum federated learning with decentralized data," IEEE J. Sel. Topics Quantum Electron., vol. 28, no. 4, Art. no. 6500110, 2022.
14. Y. Song, Y. Wu, S. Wu, D. Li, Q. Wen, S. Qin, and F. Gao, "A quantum federated learning framework for classical clients," Sci. China Phys. Mech. Astron., vol. 67, no. 5, Art. no. 250311, 2024, doi: 10.1007/s11433-023-2337-2.
15. M. R. Uddin, S. Shaon, R. Rahman, D. C. Nguyen, O. A. Dobre, and D. Niyato, "Quantum federated learning: A comprehensive survey," IEEE Commun. Surveys Tuts., vol. 28, 2026, doi: 10.1109/COMST.2025.3644750.
16. R. Ballester, J. Cerquides, and L. Artiles, "Quantum federated learning: A comprehensive literature review of foundations, challenges, and future directions," Quantum Mach. Intell., 2025, doi: 10.1007/s42484-025-00292-2.
17. D. Yin, A. Pananjady, M. Lam, D. Papailiopoulos, K. Ramchandran, and P. Bartlett, "Gradient diversity: A key ingredient for scalable distributed learning," in Proc. 21st Int. Conf. Artif. Intell. Statist. (AISTATS), PMLR, vol. 84, 2018, pp. 1998-2007.
18. D. A. Meyer and N. R. Wallach, "Global entanglement in multiparticle systems," J. Math. Phys., vol. 43, no. 9, p. 4273, 2002, doi: 10.1063/1.1497700.
19. S. Sim, P. D. Johnson, and A. Aspuru-Guzik, "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms," Adv. Quantum Technol., vol. 2, no. 12, Art. no. 1900070, 2019.
20. J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neural network training landscapes," Nat. Commun., vol. 9, Art. no. 4812, 2018, doi: 10.1038/s41467-018-07090-4.
21. H. Zhao, "Non-IID quantum federated learning with one-shot communication complexity," Quantum Mach. Intell., vol. 5, Art. no. 3, 2023, doi: 10.1007/s42484-022-00091-z.
22. Y. Wang, X. He, Z. Cai, R. Zhang, R. Ma, Y. Liu, and R. Buyya, "FEDCOMPASS: Federated clustered and periodic aggregation framework for hybrid classical-quantum models," arXiv:2602.03052, 2026.
23. S. I. Nanayakkara and S. R. Pokhrel, "A drift-stable quantum federated learning for intelligent services," arXiv:2607.21647, 2026.
24. M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information, 10th anniversary ed. Cambridge, U.K.: Cambridge Univ. Press, 2010.
25. G. K. Brennen, "An observable measure of entanglement for pure states of multi-qubit systems," Quantum Inf. Comput., vol. 3, no. 6, pp. 619-626, 2003, doi: 10.26421/QIC3.6-5.
26. V. Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical computations," arXiv:1811.04968, 2018.
27. H. Xiao, K. Rasul, and R. Vollgraf, "Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms," arXiv:1708.07747, 2017.
28. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.
29. J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification," Sci. Data, vol. 10, Art. no. 41, 2023, doi: 10.1038/s41597-022-01721-8.
30. J. N. Kather et al., "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study," PLOS Med., vol. 16, no. 1, Art. no. e1002730, 2019, doi: 10.1371/journal.pmed.1002730.
31. A. Krizhevsky, "Learning multiple layers of features from tiny images," Tech. Rep., Univ. of Toronto, 2009.
