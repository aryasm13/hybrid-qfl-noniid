import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqc_ansatz import get_vqc_layer, N_QUBITS


class HybridQNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # classical feature extractor: 28x28 -> 4 values
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 7 * 7, 32)
        self.fc2   = nn.Linear(32, N_QUBITS)

        # quantum layer
        self.vqc = get_vqc_layer()

        # classification head
        self.head = nn.Linear(N_QUBITS, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) * 3.14159  # scale to [-pi, pi]

        x = self.vqc(x)                          # (batch, 4) -> (batch, 4) PauliZ expectations
        return self.head(x)

    def get_quantum_params(self):
        return {k: v for k, v in self.state_dict().items() if "vqc" in k}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
