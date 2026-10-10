import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqc_ansatz import get_vqc_layer, N_QUBITS


def class_codes(num_classes, dim=N_QUBITS):
    # corners of a regular simplex in `dim` dimensions plus their opposites, all unit length
    simplex = np.eye(dim + 1) - 1 / (dim + 1)
    basis = np.linalg.svd(simplex)[2][:dim]
    corners = simplex @ basis.T
    corners /= np.linalg.norm(corners, axis=1, keepdims=True)
    return torch.tensor(np.vstack([corners, -corners])[:num_classes], dtype=torch.float32)


class HybridQNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, fixed_readout=False, twin=False, scale=5.0,
                 frozen_readout=False, pauli_readout=False):
        super().__init__()

        # classical feature extractor: 28x28 -> 4 values
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 7 * 7, 32)
        self.fc2   = nn.Linear(32, N_QUBITS)

        # quantum layer (twin: a classical layer of the same width instead of the circuit)
        if twin:
            self.vqc = nn.Sequential(nn.Linear(N_QUBITS, N_QUBITS), nn.Tanh())
        else:
            self.vqc = get_vqc_layer(pairs=pauli_readout)

        # classification head (fixed readout: frozen class observables, never trained)
        n_readings = N_QUBITS + N_QUBITS * (N_QUBITS - 1) // 2 if pauli_readout else N_QUBITS
        self.head = nn.Linear(n_readings, num_classes)
        if fixed_readout:
            with torch.no_grad():
                self.head.weight.copy_(scale * class_codes(num_classes))
                self.head.bias.zero_()
        if pauli_readout:
            # class c is read from its own Pauli-Z string: Z1..Z4, then Z1Z2, Z1Z3, ..., Z3Z4
            with torch.no_grad():
                self.head.weight.copy_(scale * torch.eye(num_classes, n_readings))
                self.head.bias.zero_()
        if fixed_readout or frozen_readout or pauli_readout:
            self.head.requires_grad_(False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) * 3.14159  # scale to [-pi, pi]

        x = self.vqc(x)                          # (batch, 4) -> (batch, 4 or 10) PauliZ expectations
        return self.head(x)

    def get_quantum_params(self):
        return {k: v for k, v in self.state_dict().items() if "vqc" in k}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
