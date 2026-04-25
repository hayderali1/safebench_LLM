"""
demo.py -- SafeBench-LLM mini benchmark.

Runs every filter on scenario S1 across three supervisor-latency
regimes and three seeds, printing a small comparison table.  Verifies
that the entire pipeline (filters, latency injection, perception
staleness, Highway-Env, OSQP solve) is wired correctly.

Usage:
    python -m safebench_llm.demo
"""
from __future__ import annotations
import time
import numpy as np

from .simulation import run_one
from .filters import (
    NoSafetyFilter, CBF_QP, ISSf_CBF, DCBF, HOCBF, MR_CBF,
    SDCBF, RobustSDCBF, PredictorFeedbackCBF, MPC_CBF, LAWS,
)


def make_filter(name: str, T_sup: float):
    """Factory: returns a fresh filter for the given name and latency."""
    safe_dist = 5.0
    if name == "F0_NoSafety":   return NoSafetyFilter(safe_dist=safe_dist)
    if name == "F1_CBF_QP":     return CBF_QP(safe_dist=safe_dist)
    if name == "F2_ISSf_CBF":   return ISSf_CBF(safe_dist=safe_dist, eps=10.0)
    if name == "F3_DCBF":       return DCBF(safe_dist=safe_dist, gamma=0.3)
    if name == "F4_HOCBF":      return HOCBF(safe_dist=safe_dist)
    if name == "F5_MR_CBF":     return MR_CBF(safe_dist=safe_dist, e_x=0.3, L_h=2.0)
    if name == "F6_SDCBF":      return SDCBF(safe_dist=safe_dist, sample_T=0.1, M_bound=40.0)
    if name == "F7_RSDCBF":     return RobustSDCBF(safe_dist=safe_dist, sample_T=0.1,
                                                   M_bound=40.0, d_max=0.5, L_d=1.0)
    if name == "F8_PF_CBF":     return PredictorFeedbackCBF(safe_dist=safe_dist, tau=T_sup)
    if name == "F9_MPC_CBF":    return MPC_CBF(safe_dist=safe_dist, N=4)
    if name == "LAWS+F1":       return LAWS(CBF_QP(safe_dist=safe_dist), tau_init=T_sup)
    raise ValueError(name)


def main():
    filter_names = [
        "F0_NoSafety", "F1_CBF_QP", "F2_ISSf_CBF", "F3_DCBF", "F4_HOCBF",
        "F5_MR_CBF", "F6_SDCBF", "F7_RSDCBF", "F8_PF_CBF", "F9_MPC_CBF",
        "LAWS+F1",
    ]
    T_sups = [0.05, 0.5, 2.0]
    seeds  = [0, 1, 2]

    header = f"{'Filter':<14} | " + " | ".join(f"Tsup={t:>4.2f}s" for t in T_sups)
    print("\n" + "=" * len(header))
    print("SafeBench-LLM mini benchmark, scenario S1 (highway-v0)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    t_start = time.time()
    for fn in filter_names:
        cells = []
        for T_sup in T_sups:
            n_coll, n_run, h_min_min = 0, 0, np.inf
            for s in seeds:
                f = make_filter(fn, T_sup)
                r = run_one("S1", f, T_sup=T_sup, tau_perc=0.1,
                            seed=s, max_steps=40)
                n_coll += int(r.collided)
                n_run  += 1
                h_min_min = min(h_min_min, r.h_min)
            cells.append(f"{n_coll}/{n_run} ({h_min_min:>+5.1f})")
        print(f"{fn:<14} | " + " | ".join(f"{c:<13}" for c in cells))

    print("-" * len(header))
    print(f"Wall time: {time.time() - t_start:.1f}s")
    print("Cell format:  collisions/runs (worst h_min over seeds)")
    print("h_min > 0  = barrier never violated;  h_min < 0 = constraint violated.\n")


if __name__ == "__main__":
    main()
