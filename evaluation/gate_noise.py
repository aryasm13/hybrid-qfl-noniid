import torch
import pennylane as qml
from models.vqc_ansatz import ansatz, N_QUBITS
from .shot_noise import reading_signs
from .entanglement import get_state_circuit


def get_noisy_circuit(p):
    # density-matrix simulation with a depolarizing channel of strength p after every gate
    # (4 RX of the encoding, 8 Rot and 8 CNOT), on each qubit the gate acts on
    def depolarize(op, **kwargs):
        for wire in op.wires:
            qml.DepolarizingChannel(p, wires=wire)

    noise_model = qml.NoiseModel({qml.noise.op_in(["RX", "Rot", "CNOT"]): depolarize})
    dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        ansatz(inputs, weights)
        return qml.density_matrix(wires=range(N_QUBITS))

    return qml.noise.add_noise(qml.transforms.decompose(circuit, gate_set={"RX", "Rot", "CNOT"}), noise_model)


def readout_matrix(q):
    # every measured bit is reported flipped with probability q
    flip = torch.tensor([[1 - q, q], [q, 1 - q]], dtype=torch.float64)
    matrix = flip
    for _ in range(N_QUBITS - 1):
        matrix = torch.kron(matrix, flip)
    return matrix


def noise_accuracies(model, data_loader, gate_levels, readout_levels, pairs=False):
    # test accuracy for every (gate noise p, readout error q) with exact expectations (no shots),
    # and the mean fidelity <psi|rho|psi> between the noisy and the noiseless state for every p
    ideal = get_state_circuit()
    noisy = {p: get_noisy_circuit(p) for p in gate_levels if p > 0}
    readouts = {q: readout_matrix(q) for q in readout_levels}
    signs = reading_signs(pairs).double()
    weights = model.vqc.weights.detach()
    correct = {(p, q): 0 for p in gate_levels for q in readout_levels}
    fidelity = {p: 0.0 for p in gate_levels}
    total = 0
    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            angles = model.encode(data)
            psi = ideal(angles, weights)
            for p in gate_levels:
                if p > 0:
                    rho = noisy[p](angles, weights)
                    probs = torch.diagonal(rho, dim1=1, dim2=2).real
                    fidelity[p] += torch.einsum("bi,bij,bj->b", psi.conj().to(rho.dtype), rho, psi.to(rho.dtype)).real.sum().item()
                else:
                    probs = psi.abs() ** 2
                    fidelity[p] += len(psi)
                for q in readout_levels:
                    readings = probs.double() @ readouts[q] @ signs
                    correct[(p, q)] += (model.head(readings.float()).argmax(dim=1) == targets).sum().item()
            total += len(targets)
    return ({key: value / total for key, value in correct.items()},
            {p: value / total for p, value in fidelity.items()})
