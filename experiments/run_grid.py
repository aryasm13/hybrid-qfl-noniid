import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RUNNERS = {
    "fedavg":       "run_qfl_fedavg.py",
    "fedprox":      "run_fedprox.py",
    "fedprox_full": "run_fedprox.py",
    "weighted":     "run_weighted_agg.py",
    "classical":    "run_classical_baseline.py",
    "fco":          "run_qfl_fedavg.py",
    "logit_adjust": "run_logit_adjust.py",
    "twin":         "run_qfl_fedavg.py",
    "frozen":       "run_qfl_fedavg.py",
    "pauli":        "run_qfl_fedavg.py",
    "twin_fco":     "run_qfl_fedavg.py",
}


def build_job(strategy, split, seed, args):
    if split == "iid":
        partition, alpha = "iid", 0.5
    else:
        partition, alpha = "dirichlet", float(split.replace("dir", ""))

    cmd = [sys.executable, os.path.join("experiments", RUNNERS[strategy]),
           "--partition", partition, "--alpha", str(alpha),
           "--clients", str(args.clients), "--rounds", str(args.rounds),
           "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
           "--lr", str(args.lr), "--seed", str(seed)]

    tail = f"c{args.clients}_e{args.epochs}_lr{args.lr}_s{seed}.csv"
    if strategy == "fedavg":
        name = f"qfl_fedavg_{partition}_a{alpha}_{tail}"
    elif strategy == "classical":
        name = f"classical_{partition}_a{alpha}_{tail}"
    elif strategy == "weighted":
        cmd += ["--lam", str(args.lam)]
        name = f"qfl_weighted_{partition}_a{alpha}_lam{args.lam}_{tail}"
    elif strategy == "fco":
        cmd.append("--fixed_readout")
        name = f"qfl_fco_{partition}_a{alpha}_{tail}"
    elif strategy == "logit_adjust":
        cmd += ["--tau", str(args.tau)]
        name = f"qfl_logitadj_{partition}_a{alpha}_tau{args.tau}_{tail}"
    elif strategy == "twin":
        cmd.append("--twin")
        name = f"qfl_twin_{partition}_a{alpha}_{tail}"
    elif strategy == "frozen":
        cmd.append("--frozen_readout")
        name = f"qfl_frozen_{partition}_a{alpha}_{tail}"
    elif strategy == "pauli":
        cmd.append("--pauli_readout")
        name = f"qfl_pauli_{partition}_a{alpha}_{tail}"
    elif strategy == "twin_fco":
        cmd += ["--twin", "--fixed_readout"]
        name = f"qfl_twin_fco_{partition}_a{alpha}_{tail}"
    else:
        cmd += ["--mu", str(args.mu)]
        prefix = "qfl_fedprox"
        if strategy == "fedprox_full":
            cmd.append("--full_prox")
            prefix = "qfl_fedprox_full"
        name = f"{prefix}_{partition}_a{alpha}_mu{args.mu}_{tail}"

    return cmd, os.path.join(ROOT, "results", "tables", name)


def is_done(csv_path, rounds):
    if not os.path.exists(csv_path):
        return False
    with open(csv_path) as f:
        return sum(1 for _ in f) - 1 >= rounds


def run_job(job, threads):
    cmd, csv_path = job
    name = os.path.basename(csv_path)[:-4]
    log_path = os.path.join(ROOT, "results", "logs", name + ".log")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    minutes = (time.time() - start) / 60

    if code == 0:
        print(f"done    {name}  ({minutes:.1f} min)", flush=True)
    else:
        print(f"FAILED  {name}  (exit {code}, see results/logs/{name}.log)", flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["fedavg", "fedprox", "fedprox_full", "weighted"],
                        choices=list(RUNNERS))
    parser.add_argument("--splits",     nargs="+", default=["dir0.1", "dir0.5", "iid"])
    parser.add_argument("--seeds",      nargs="+", type=int, default=[42])
    parser.add_argument("--clients",    type=int,   default=10)
    parser.add_argument("--rounds",     type=int,   default=20)
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.003)
    parser.add_argument("--mu",         type=float, default=0.01)
    parser.add_argument("--lam",        type=float, default=0.5)
    parser.add_argument("--tau",        type=float, default=1.0)
    parser.add_argument("--workers",    type=int,   default=1)
    parser.add_argument("--threads",    type=int,   default=4)
    parser.add_argument("--dry_run",    action="store_true")
    args = parser.parse_args()

    jobs = [build_job(st, sp, seed, args)
            for seed in args.seeds for sp in args.splits for st in args.strategies]
    todo = [job for job in jobs if not is_done(job[1], args.rounds)]
    print(f"{len(jobs)} runs in the grid, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run with {args.workers} worker(s)\n")

    if args.dry_run:
        for cmd, csv_path in todo:
            print(" ".join(cmd[1:]))
        sys.exit(0)

    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(lambda job: run_job(job, args.threads), todo))

    failed = sum(1 for code in codes if code != 0)
    print(f"\nFinished {len(todo)} runs in {(time.time() - start) / 60:.1f} min, {failed} failed")
