from .fedavg import train_one_round, aggregate_fedavg
from .quantum_fedprox import train_fedprox
from .weighted_aggregation import aggregate_weighted

__all__ = [
    "train_one_round",
    "aggregate_fedavg",
    "train_fedprox",
    "aggregate_weighted"
]
