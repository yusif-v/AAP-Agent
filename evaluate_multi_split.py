#!/usr/bin/env python3
"""
Phase 1 measurement harness: runs the training pipeline (train_automl.py) across
multiple data/train_XX splits and reports mean +/- std held-out AUC, scored against
each split's ground-truth solution.csv. This is the "scoreboard" number - a change
only counts if it improves this across most splits by more than noise (see std).

Usage:
    python evaluate_multi_split.py [--splits N] [--tune] [--experiment ensemble]
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO / "submission/skills/model-training/scripts/train_automl.py"
DATA_DIR = REPO / "data"
PYTHON = "/usr/bin/python3"


def run_split(split_dir, extra_args):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(split_dir / "train.csv", tmp / "train.csv")
        shutil.copy(split_dir / "test.csv", tmp / "test.csv")
        cmd = [str(PYTHON), str(TRAIN_SCRIPT)] + extra_args
        result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-2000:]}")
            return None
        sub = pd.read_csv(tmp / "final_submission.csv")
        sol = pd.read_csv(split_dir / "solution.csv")
        m = sol.merge(sub, on="row_id", suffixes=("_true", "_pred"))
        return roc_auc_score(m["target_true"], m["target_pred"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5, help="number of train_XX splits to evaluate")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--experiment", default="ensemble")
    args = ap.parse_args()

    split_dirs = sorted(p for p in DATA_DIR.glob("train_*") if p.is_dir())[: args.splits]
    extra_args = ["--experiment", args.experiment] + (["--tune"] if args.tune else [])

    scores = []
    for split_dir in split_dirs:
        print(f"[{split_dir.name}] running ({' '.join(extra_args)})...")
        auc = run_split(split_dir, extra_args)
        if auc is not None:
            print(f"[{split_dir.name}] held-out AUC = {auc:.5f}")
            scores.append((split_dir.name, auc))

    if not scores:
        print("No successful runs.")
        sys.exit(1)

    aucs = [s for _, s in scores]
    mean = sum(aucs) / len(aucs)
    std = (sum((a - mean) ** 2 for a in aucs) / len(aucs)) ** 0.5

    print("\n--- Scoreboard ---")
    for name, auc in scores:
        print(f"  {name}: {auc:.5f}")
    print(f"mean={mean:.5f}  std={std:.5f}  n={len(aucs)}")


if __name__ == "__main__":
    main()
