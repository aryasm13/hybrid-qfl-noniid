from itertools import combinations
import torch
import pennylane as qml
from models.vqc_ansatz import ansatz, N_QUBITS
from .shot_noise import reading_signs


def get_state_circuit():
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        ansatz(inputs, weights)
        return qml.state()

    return circuit


def state_properties(model, data_loader):
    # averages over the test images of properties of the circuit state |psi(x)>:
    # meyer_wallach: entanglement, 0 for a product state and at most 1 (Q = 1 - mean squared Bloch vector length)
    # z_norm: mean squared Z reading, which can never be above 1 - Q, so entanglement shrinks the readings
    # zz_connected: mean |<ZiZj> - <Zi><Zj>| over the 6 pairs, 0 for a product state
    # entropy: Shannon entropy in bits of the 16 measurement outcomes, 0 for one bitstring, 4 for all equal
    circuit = get_state_circuit()
    weights = model.vqc.weights.detach()
    signs = reading_signs(pairs=True).double()
    pairs = list(combinations(range(N_QUBITS), 2))
    sums = {"meyer_wallach": 0.0, "z_norm": 0.0, "zz_connected": 0.0, "entropy": 0.0}
    total = 0
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            psi = circuit(model.encode(data), weights)
            amplitudes = psi.reshape(-1, *[2] * N_QUBITS)
            purity = 0
            for k in range(N_QUBITS):
                # reduced state of qubit k, the other three qubits traced out
                m = amplitudes.movedim(k + 1, 1).reshape(len(psi), 2, -1)
                rho = m @ m.conj().transpose(1, 2)
                purity = purity + (rho.abs() ** 2).sum(dim=(1, 2))
            probs = psi.abs() ** 2
            readings = probs @ signs
            z, zz = readings[:, :N_QUBITS], readings[:, N_QUBITS:]
            connected = torch.stack([zz[:, n] - z[:, i] * z[:, j] for n, (i, j) in enumerate(pairs)], dim=1)
            sums["meyer_wallach"] += (2 * (1 - purity / N_QUBITS)).sum().item()
            sums["z_norm"] += (z ** 2).mean(dim=1).sum().item()
            sums["zz_connected"] += connected.abs().mean(dim=1).sum().item()
            sums["entropy"] += -(probs * torch.log2(probs.clamp(min=1e-12))).sum().item()
            total += len(psi)
    return {name: value / total for name, value in sums.items()}
