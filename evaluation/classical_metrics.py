import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(model, data_loader, device="cpu"):
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            total_loss += criterion(outputs, targets).item() * len(targets)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += len(targets)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }
