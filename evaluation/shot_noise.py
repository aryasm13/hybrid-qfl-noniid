from itertools import combinations
import torch
from models.vqc_ansatz import get_vqc_layer, N_QUBITS


def reading_signs(pairs=False):
    # value (+1 or -1) of every Z reading for each of the 16 bitstrings; wire 0 is the leftmost bit
    bits = torch.tensor([[(b >> (N_QUBITS - 1 - i)) & 1 for i in range(N_QUBITS)] for b in range(2 ** N_QUBITS)])
    signs = 1 - 2 * bits.float()
    if pairs:
        signs = torch.cat([signs] + [signs[:, [i]] * signs[:, [j]] for i, j in combinations(range(N_QUBITS), 2)], dim=1)
    return signs


def shot_accuracies(model, data_loader, shots_list, pairs=False, seed=0):
    # test accuracy when every reading is estimated from a finite number of shots,
    # like on real hardware: all qubits are measured `shots` times and every Z reading
    # (and every Z_i Z_j pair) is computed from those same measured bitstrings
    probs_layer = get_vqc_layer(probs=True)
    probs_layer.load_state_dict(model.vqc.state_dict())
    signs = reading_signs(pairs)
    generator = torch.Generator().manual_seed(seed)
    correct = {shots: 0 for shots in shots_list}
    total = 0
    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            probs = probs_layer(model.encode(data)).float().clamp(min=0)
            for shots in shots_list:
                picks = torch.multinomial(probs, shots, replacement=True, generator=generator)
                counts = torch.zeros_like(probs).scatter_add_(1, picks, torch.ones_like(picks, dtype=probs.dtype))
                readings = (counts / shots) @ signs
                correct[shots] += (model.head(readings).argmax(dim=1) == targets).sum().item()
            total += len(targets)
    return {shots: correct[shots] / total for shots in shots_list}
