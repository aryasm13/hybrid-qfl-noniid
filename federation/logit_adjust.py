import torch
import torch.nn as nn


def train_logit_adjusted(model, train_loader, log_prior, tau=1.0, epochs=1, lr=0.001, device="cpu"):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # adding log(class share) means classes this client never sees are not pushed down
    shift = tau * log_prior.to(device)

    for _ in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data) + shift, targets)
            loss.backward()
            optimizer.step()

    return {k: v.cpu().clone() for k, v in model.state_dict().items()}
