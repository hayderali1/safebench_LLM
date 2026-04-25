"""
benchmark.py -- SafeBench-LLM full benchmark grid runner.

Sweeps  filter x scenario x T_sup x tau_perc x seed  and writes a tidy
CSV / parquet with one row per run.  This is the artifact the paper
analyses; figures and tables in the manuscript are built from it.

Usage (from the project root):
    python -m safebench_llm.benchmark --quick    # ~2 minutes, demo grid
    python -m safebench_llm.benchmark            # full paper grid
"""
from __future__ import annotations
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .simulation import run_one
from .filters import (
    NoSafetyFilter, CBF_QP, ISSf_CBF, DCBF, HOCBF, MR_CBF,
    SDCBF, RobustSDCBF, PredictorFeedbackCBF, MPC_CBF, LAWS,
)


# Filter factories -- one per row, fresh instance per cell so that
# stateful filters (PF-CBF.u_held, LAWS.timestamps) start clean.
def _factories(T_sup: float):
    safe_dist = 5.0
    return {
        "F0_NoSafety":   lambda: NoSafetyFilter(safe_dist=safe_dist),
        "F1_CBF_QP":     lambda: CBF_QP(safe_dist=safe_dist, alpha=2.0),
        "F2_ISSf_CBF":   lambda: ISSf_CBF(safe_dist=safe_dist, alpha=2.0, eps=10.0),
        "F3_DCBF":       lambda: DCBF(safe_dist=safe_dist, gamma=0.3),
        "F4_HOCBF":      lambda: HOCBF(safe_dist=safe_dist, alpha1=1.5, alpha2=1.5),
        "F5_MR_CBF":     lambda: MR_CBF(safe_dist=safe_dist, alpha=2.0,
                                        e_x=0.3, L_h=2.0),
        "F6_SDCBF":      lambda: SDCBF(safe_dist=safe_dist, alpha=2.0,
                                       sample_T=0.1, M_bound=40.0),
        "F7_RSDCBF":     lambda: RobustSDCBF(safe_dist=safe_dist, alpha=2.0,
                                             sample_T=0.1, M_bound=40.0,
                                             d_max=0.5, L_d=1.0),
        "F8_PF_CBF":     lambda: PredictorFeedbackCBF(safe_dist=safe_dist,
                                                      alpha=2.0,
                                                      tau=max(T_sup, 0.05)),
        "F9_MPC_CBF":    lambda: MPC_CBF(safe_dist=safe_dist, N=4),
        "LAWS+CBF_QP":   lambda: LAWS(CBF_QP(safe_dist=safe_dist, alpha=2.0),
                                      tau_init=max(T_sup, 0.05),
                                      L_h=2.0, e_x=0.3),
    }


def run_grid(filters, scenarios, T_sups, tau_percs, seeds,
             max_steps: int = 80, verbose: bool = True):
    """Returns a pandas DataFrame, one row per (filter, scenario,
    T_sup, tau_perc, seed)."""
    rows = []
    total = (len(filters) * len(scenarios) * len(T_sups)
             * len(tau_percs) * len(seeds))
    done, t0 = 0, time.time()
    for fname in filters:
        for scen in scenarios:
            for T_sup in T_sups:
                for tau_perc in tau_percs:
                    for seed in seeds:
                        f = _factories(T_sup)[fname]()
                        r = run_one(scen, f, T_sup=T_sup,
                                    tau_perc=tau_perc, seed=seed,
                                    max_steps=max_steps)
                        rows.append({
                            "filter":   fname,
                            "scenario": scen,
                            "T_sup":    T_sup,
                            "tau_perc": tau_perc,
                            "seed":     seed,
                            "collided": r.collided,
                            "viol":     r.constraint_violated,
                            "h_min":    r.h_min,
                            "interv":   r.avg_intervention,
                            "infeas":   r.qp_infeas_rate,
                            "solve_ms": r.avg_solve_ms,
                            "steps":    r.steps,
                            "avg_v":    r.avg_speed,
                        })
                        done += 1
                        if verbose and done % 10 == 0:
                            elapsed = time.time() - t0
                            rate = done / elapsed if elapsed > 0 else 0
                            eta  = (total - done) / rate if rate > 0 else 0
                            print(f"  {done:4d}/{total} runs   "
                                  f"rate={rate:4.1f}/s   eta={eta:5.1f}s",
                                  flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (filter, scenario, T_sup): mean collision rate, mean
    h_min, mean intervention.  This is what the headline tables in the
    paper plot."""
    g = df.groupby(["filter", "scenario", "T_sup"])
    out = g.agg(
        n_seeds  = ("seed",     "count"),
        coll_rate= ("collided", "mean"),
        viol_rate= ("viol",     "mean"),
        h_min_min= ("h_min",    "min"),
        h_min_med= ("h_min",    "median"),
        interv   = ("interv",   "mean"),
        infeas   = ("infeas",   "mean"),
        solve_ms = ("solve_ms", "mean"),
    ).reset_index()
    return out


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small grid (~2 min), for smoke / demo")
    ap.add_argument("--out", default="/mnt/user-data/outputs/benchmark_results.csv",
                    help="output CSV path")
    ap.add_argument("--max-steps", type=int, default=80)
    args = ap.parse_args()

    if args.quick:
        filters   = ["F0_NoSafety", "F1_CBF_QP", "F4_HOCBF",
                     "F6_SDCBF", "F8_PF_CBF", "LAWS+CBF_QP"]
        scenarios = ["S1", "S4"]
        T_sups    = [0.05, 0.5, 2.0]
        tau_percs = [0.0]
        seeds     = [0, 1, 2]
    else:
        filters   = list(_factories(0.1).keys())
        scenarios = ["S1", "S2", "S4", "S5"]
        T_sups    = [0.05, 0.5, 2.0]
        tau_percs = [0.0, 0.2]
        seeds     = [0, 1, 2]

    print(f"\nSafeBench-LLM benchmark grid")
    print(f"  filters   = {len(filters)} ({filters})")
    print(f"  scenarios = {scenarios}")
    print(f"  T_sup     = {T_sups}")
    print(f"  tau_perc  = {tau_percs}")
    print(f"  seeds     = {seeds}")
    total = (len(filters) * len(scenarios) * len(T_sups)
             * len(tau_percs) * len(seeds))
    print(f"  total     = {total} runs\n")

    t0 = time.time()
    df = run_grid(filters, scenarios, T_sups, tau_percs, seeds,
                  max_steps=args.max_steps)
    elapsed = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = summarise(df)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nFinished in {elapsed:.1f}s")
    print(f"  raw rows: {len(df)}  ->  {out_path}")
    print(f"  summary:  {len(summary)}  ->  {summary_path}\n")

    # ---- pretty terminal summary ----
    print("=" * 80)
    print("Collision rate by filter x T_sup, averaged across all scenarios:")
    print("=" * 80)
    pivot = (df.groupby(["filter", "T_sup"])["collided"]
               .mean().unstack().fillna(0.0))
    print(pivot.to_string(float_format=lambda x: f"{100*x:5.1f}%"))
    print()


if __name__ == "__main__":
    main()
