#!/usr/bin/env python3
"""Standard entry point.

    python run.py <dataset_dir> --out <output_dir>
    python run.py --train train.csv --test test.csv --meta metadata.json --out out/

The dataset directory is expected to hold a training table, an optional test
table and a metadata JSON (file names configurable in automl/config.py).
Task, target, number of clusters and id column may also be given on the
command line and override the metadata.
"""
import time

T_START = time.time()  # the budget counts from process start, imports included

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automl import config  # noqa: E402
from automl.pipeline import run_path  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="dataset directory (or a single CSV)")
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--meta")
    ap.add_argument("--task", choices=["classification", "regression", "clustering"])
    ap.add_argument("--target")
    ap.add_argument("--n-clusters", type=int)
    ap.add_argument("--id-col")
    ap.add_argument("--out", default="output")
    ap.add_argument("--time-limit", type=float, default=config.TIME_LIMIT_SECONDS)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-watchdog", action="store_true",
                    help="do not terminate the process at the time limit")
    a = ap.parse_args(argv)
    if not (a.path or a.train):
        ap.error("give a dataset directory or --train")
    rep = run_path(a.path, out_dir=a.out, time_limit=a.time_limit, verbose=not a.quiet,
                   train=a.train, test=a.test, meta=a.meta, task=a.task, target=a.target,
                   n_clusters=a.n_clusters, id_col=a.id_col, hard_limit=not a.no_watchdog,
                   t_start=T_START)
    return 1 if rep.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
