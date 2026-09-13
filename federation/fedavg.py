import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_round(model, train_loader, epochs=1, lr=0.001, device="cpu"):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), targets)
            loss.backward()
            optimizer.step()

    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def aggregate_fedavg(client_weights, sample_counts):
    total = sum(sample_counts)
    global_weights = {}
    for key in client_weights[0].keys():
        global_weights[key] = sum(
            (count / total) * client_weights[i][key].float()
            for i, count in enumerate(sample_counts)
        )
    return global_weights
