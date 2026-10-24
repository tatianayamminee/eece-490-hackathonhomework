#!/usr/bin/env python3
"""Local checker: run every dataset under a directory and score the outputs.

    python tools/check.py data/synthetic [--time-limit 60]

Scores: classification -> accuracy, macro-F1; regression -> RMSE, R2;
clustering -> adjusted Rand index against solution.csv. Also reports wall time
and whether profile.json / predictions.csv exist. Replace the metrics once the
official scoring rules are known.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             mean_squared_error, r2_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    a = ap.parse_args()
    rows = []
    for name in sorted(os.listdir(a.data_dir)):
        d = os.path.join(a.data_dir, name)
        if not os.path.isdir(d) or (a.only and name not in a.only):
            continue
        out = os.path.join(a.out, name)
        t0 = time.time()
        try:  # the limit is enforced here exactly as an assessor would
            proc = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), d, "--out",
                                   out, "--time-limit", str(a.time_limit), "--quiet"],
                                  capture_output=True, text=True, timeout=a.time_limit)
            rc, err = proc.returncode, proc.stderr
        except subprocess.TimeoutExpired as e:
            rc, err = "TIMEOUT", (e.stderr or b"").decode(errors="replace") \
                if isinstance(e.stderr, bytes) else (e.stderr or "")
        el = time.time() - t0
        row = {"dataset": name, "seconds": round(el, 1), "rc": rc,
               "within_limit": el <= a.time_limit and rc != "TIMEOUT",
               "profile": os.path.exists(os.path.join(out, "profile.json")),
               "preds": os.path.exists(os.path.join(out, "predictions.csv"))}
        if rc or not row["preds"]:
            row["stderr"] = err[-800:]
        try:
            rep = json.load(open(os.path.join(out, "run_report.json")))
            row["chosen"] = rep.get("chosen")
            row["cv"] = rep.get("cv_score")
        except Exception:
            pass
        sol_p = os.path.join(d, "solution.csv")
        meta = json.load(open(os.path.join(d, "metadata.json")))
        if row["preds"] and os.path.exists(sol_p):
            pred = pd.read_csv(os.path.join(out, "predictions.csv"))
            sol = pd.read_csv(sol_p)
            if "row_id" in pred.columns and "row_id" in sol.columns:
                m = sol.merge(pred, on="row_id", how="left")
                y, p = m.iloc[:, 1], m["prediction"]
            else:
                y, p = sol.iloc[:, -1], pred["prediction"]
            task = meta["task"]
            if task == "classification":
                y, p = y.astype(str), p.astype(str)
                row["accuracy"] = round(accuracy_score(y, p), 4)
                row["f1_macro"] = round(f1_score(y, p, average="macro"), 4)
            elif task == "regression":
                row["rmse"] = round(float(np.sqrt(mean_squared_error(y, p))), 4)
                row["r2"] = round(r2_score(y, p), 4)
            else:
                row["ari"] = round(adjusted_rand_score(y, p), 4)
        rows.append(row)
        print(json.dumps(row), flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + df.drop(columns=["stderr"], errors="ignore").to_string(index=False))
    ok = bool(df["within_limit"].all() and df["preds"].all() and df["profile"].all()
              and (df["rc"] == 0).all())
    print(f"\n{'PASS' if ok else 'FAIL'}: {int(df['within_limit'].sum())}/{len(df)} datasets "
          f"within {a.time_limit:.0f}s with predictions and profile, "
          f"{int((df['rc'] == 0).sum())}/{len(df)} without modelling errors")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
