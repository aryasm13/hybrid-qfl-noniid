import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(model, data_loader, device="cpu"):
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            total_loss += criterion(outputs, targets).item() * len(targets)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += len(targets)
            preds.append(outputs.argmax(dim=1).cpu())
            labels.append(targets.cpu())

    metrics = {
        "loss": total_loss / total,
        "accuracy": correct / total
    }
    # accuracy on each class on its own, shows which classes the global model loses under skew
    preds, labels = torch.cat(preds), torch.cat(labels)
    for c in range(outputs.shape[1]):
        metrics[f"acc_c{c}"] = (preds[labels == c] == c).float().mean().item()
    return metrics
