import pennylane as qml
from pennylane import numpy as np


N_QUBITS = 4
N_LAYERS = 2


def get_vqc_layer():
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    weight_shape = {"weights": (N_LAYERS, N_QUBITS, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shape)
