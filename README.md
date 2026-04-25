# SafeBench-LLM — reference implementation

Reference implementation for the paper *SafeBench-LLM: Benchmarking Safety
Filters for Autonomous Driving Under LLM/VLM-Supervisor Latency*. CPU-only,
Python-only, MIT-licensed. Runs end-to-end on a laptop in minutes.

## Layout

```
safebench_llm/
├── filters/                    # all 9 published filters + LAWS
│   ├── base.py                 # SafetyFilter ABC, common cbf_terms helper
│   ├── f01_cbf_qp.py           # F0 NoSafety, F1 CBF-QP (Ames et al. 2017)
│   ├── f23_isscbf_dcbf.py      # F2 ISSf-CBF, F3 DCBF
│   ├── f45_hocbf_mrcbf.py      # F4 HOCBF, F5 MR-CBF
│   ├── f678_sdcbf_pfcbf.py     # F6 SD-CBF, F7 Robust SD-CBF, F8 PF-CBF
│   └── f9_mpc_laws.py          # F9 MPC-CBF, LAWS wrapper
├── utils/dynamics.py           # kinematic bicycle, RK4 integration
├── simulation.py               # closed-loop harness, latency injection
├── benchmark.py                # full grid runner, CSV output
├── make_figures.py             # matplotlib figures from CSV
└── demo.py                     # 3-minute smoke benchmark
```

## Quickstart

```bash
pip install gymnasium highway-env cvxpy osqp pandas matplotlib

# 3-minute smoke benchmark, ~108 runs:
python -m safebench_llm.benchmark --quick

# Regenerate figures:
python -m safebench_llm.make_figures
```

Outputs land in `/mnt/user-data/outputs/`:
- `benchmark_results.csv`           — one row per run
- `benchmark_results_summary.csv`   — collision/intervention/etc. by cell
- `figures/fig_collision_vs_tsup.png`
- `figures/fig_hmin_heatmap.png`
- `figures/fig_pareto.png`

## The 9 filters

| ID | Filter                | Source paper                                                |
|----|-----------------------|-------------------------------------------------------------|
| F0 | No-Safety             | (negative baseline)                                         |
| F1 | CBF-QP                | Ames, Xu, Grizzle, Tabuada, IEEE TAC 2017                   |
| F2 | ISSf-CBF              | Kolathaya & Ames, IEEE L-CSS 2018                           |
| F3 | DCBF                  | LA-RL (Shu et al. 2025); classical form                     |
| F4 | HOCBF                 | Xiao, Cassandras, Belta, Springer 2023                      |
| F5 | MR-CBF                | Cosner et al., arXiv:2104.14030 (2021)                      |
| F6 | SD-CBF                | Breeden, Garg, Panagou, IEEE L-CSS 2021                     |
| F7 | Robust SD-CBF         | Oruganti, Naghizadeh, arXiv:2309.07022 (2023)               |
| F8 | Predictor-Feedback    | Jankovic 2018; Molnar et al. 2026; Kim et al. 2025          |
| F9 | MPC-CBF               | Aksun-Guvenc et al., Electronics 2025                       |
| —  | LAWS+any              | This paper. Composes Breeden + Cosner + Kim                 |

The mathematical statement and citation for each filter are in
`safety_filters_report.docx`.

## Initial results

See `initial_results_report.docx`. Headline finding from the quick grid:
**LAWS+CBF-QP is the only filter to maintain a positive worst-case barrier
value (h_min > 0) at every supervisor latency from 0.05 s to 2.0 s on the
adversarial cut-in scenario S4.** F8 Predictor-Feedback CBF degrades from
h_min = -4.0 at 0.05 s to h_min = -55.9 at 2.0 s on the same scenario,
because its constant-velocity obstacle prediction is invalidated by the
adversarial brake — exactly the empirical fragility hypothesised in §4.1
RQ2 of the paper.
