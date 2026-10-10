from itertools import combinations
import pennylane as qml
from pennylane import numpy as np


N_QUBITS = 4
N_LAYERS = 2


def get_vqc_layer(pairs=False):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        readings = [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
        if pairs:
            # two-qubit parities Z_i Z_j, measured from the same computational-basis shots
            readings += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in combinations(range(N_QUBITS), 2)]
        return readings

    weight_shape = {"weights": (N_LAYERS, N_QUBITS, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shape)
