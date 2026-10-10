import torch


def client_drift(global_state, client_weights, sample_counts):
    # for every layer (conv1, conv2, fc1, fc2, vqc, head):
    # drift = average size of the client moves this round, relative to the size of the layer
    # agree = size of the averaged move / average size of the client moves
    #         (1 = all clients move the same way, near 0 = their moves cancel out in FedAvg)
    total = sum(sample_counts)
    stats = {}
    for layer in dict.fromkeys(key.split(".")[0] for key in global_state):
        keys = [key for key in global_state if key.split(".")[0] == layer]
        start = torch.cat([global_state[key].flatten().float() for key in keys])
        moves = [torch.cat([w[key].flatten().float() for key in keys]) - start for w in client_weights]
        mean_move = sum((n / total) * move for n, move in zip(sample_counts, moves))
        mean_size = sum((n / total) * move.norm().item() for n, move in zip(sample_counts, moves))
        stats[f"drift_{layer}"] = mean_size / start.norm().item()
        stats[f"agree_{layer}"] = mean_move.norm().item() / mean_size if mean_size > 0 else float("nan")
    return stats
