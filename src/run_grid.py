
"""Resume-safe runner for the ablation grid: python -m src.run_grid --models M2 M3 --folds 1 --seeds 0 [--part 1/3]"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from .config import get_paths

       
# Folder name run_ablation creates for a job (mirrors its tag rule), used to detect finished runs.
def run_name(model, fold, seed, lam):
    return f"{model}_fold{fold}_seed{seed}" + (f"_lam{lam}" if lam is not None else "")


# Builds the job list, takes this worker's share, and runs each unfinished job as a subprocess.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M1", "M2", "M3", "M4"])
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lams", nargs="+", type=float, default=None, help="lambda_turnover values, applied to M4 only")
    ap.add_argument("--part", default="1/1", help="k/n: this worker takes every n-th job starting at k")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs without running them")
    a = ap.parse_args()

    jobs = []
    for fold, seed, model in itertools.product(a.folds, a.seeds, a.models):
        for lam in (a.lams if (model == "M4" and a.lams) else [None]):
            jobs.append((model, fold, seed, lam))
    k, n = map(int, a.part.split("/"))
    jobs = jobs[k - 1::n]

    root, failed = Path(get_paths()["metrics"]), []
    for i, (model, fold, seed, lam) in enumerate(jobs, 1):
        name = run_name(model, fold, seed, lam)
        if (root / name / "metrics.json").exists():
            print(f"[{i}/{len(jobs)}] skip (done): {name}")
            continue
        cmd = [sys.executable, "-m", "src.run_ablation", "--model", model, "--fold", str(fold), "--seed", str(seed)]
        if lam is not None:
            cmd += ["--lam", str(lam)]
        print(f"[{i}/{len(jobs)}] run: {name}", flush=True)
        if a.dry_run:
            continue
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
            print("  FAILED:", name, flush=True)
    print("finished; failed jobs:", failed if failed else "none")


if __name__ == "__main__":
    main()     