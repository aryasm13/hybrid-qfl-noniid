import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results/plots", exist_ok=True)

errors = []

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK]   {label}")
        return result
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        errors.append(label)
        return None


print("\n=== Week 1 Verification ===\n")

print("--- Imports ---")
check("pennylane",   lambda: __import__("pennylane"))
check("torch",       lambda: __import__("torch"))
check("flwr",        lambda: __import__("flwr"))
check("numpy",       lambda: __import__("numpy"))
check("matplotlib",  lambda: __import__("matplotlib"))

try:
    import pennylane as qml
    print(f"     PennyLane {qml.__version__}")
except Exception:
    pass

print("\n--- Dataset Download ---")
from data.utils import load_dataset, visualize_partition

train = check("FashionMNIST train", lambda: load_dataset("fashionmnist")[0])
test  = check("FashionMNIST test",  lambda: load_dataset("fashionmnist")[1])

print("\n--- IID Partition ---")
from data.iid_partition import iid_partition

iid = check("IID partition (10 clients)", lambda: iid_partition(train, n_clients=10))
if iid:
    check("IID: 10 client keys",        lambda: None if len(iid) == 10 else (_ for _ in ()).throw(AssertionError()))
    all_idx = [i for lst in iid.values() for i in lst]
    check("IID: no duplicate indices",   lambda: None if len(all_idx) == len(set(all_idx)) else (_ for _ in ()).throw(AssertionError()))

print("\n--- Dirichlet Label-Skew Partition ---")
from data.label_skew import dirichlet_partition, compute_emd_proxy

dir_05 = check("Dirichlet alpha=0.5", lambda: dirichlet_partition(train, n_clients=10, alpha=0.5))
dir_01 = check("Dirichlet alpha=0.1", lambda: dirichlet_partition(train, n_clients=10, alpha=0.1))

if dir_05 and dir_01:
    emd_iid = check("EMD proxy (IID reference)", lambda: compute_emd_proxy(iid, train))
    emd_05  = check("EMD proxy alpha=0.5",       lambda: compute_emd_proxy(dir_05, train))
    emd_01  = check("EMD proxy alpha=0.1",       lambda: compute_emd_proxy(dir_01, train))

    if emd_iid is not None and emd_01 is not None:
        check("EMD: alpha=0.1 > IID", lambda: None if emd_01 > emd_iid else (_ for _ in ()).throw(AssertionError(f"{emd_01:.4f} should > {emd_iid:.4f}")))

print("\n--- Quantity Skew Partition ---")
from data.quantity_skew import quantity_skew_partition

qty = check("Quantity skew sigma=1.0", lambda: quantity_skew_partition(train, n_clients=10, sigma=1.0))

print("\n--- Visualization ---")
if dir_05:
    check(
        "Partition plot saved",
        lambda: visualize_partition(dir_05, train, title="Dirichlet alpha=0.5", save_path="results/plots/label_skew_0.5.png")
    )

print()
if errors:
    print(f"[!] {len(errors)} check(s) failed: {errors}")
    sys.exit(1)
else:
    print("All Week 1 checks passed. Ready for Week 2.")
