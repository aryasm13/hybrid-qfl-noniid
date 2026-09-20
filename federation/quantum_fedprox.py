import copy
import torch
import torch.nn as nn


def train_fedprox(model, global_state, train_loader, mu=0.01,
                  epochs=1, lr=0.001, quantum_only=True, device="cpu"):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            task_loss = criterion(model(data), targets)

            # proximal term: penalise drift from global model
            prox = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if quantum_only and "vqc" not in name:
                    continue
                global_val = global_state[name].to(device)
                prox = prox + ((param - global_val) ** 2).sum()

            loss = task_loss + (mu / 2) * prox
            loss.backward()
            optimizer.step()

    return {k: v.cpu().clone() for k, v in model.state_dict().items()}
