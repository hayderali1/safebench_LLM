"""
make_figures.py -- Generate the headline figures from benchmark_results.csv.

Outputs PNG files into /mnt/user-data/outputs/figures/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_collision_vs_tsup(df, out_dir):
    """One curve per filter: collision rate vs supervisor latency."""
    fig, axes = plt.subplots(1, df["scenario"].nunique(),
                             figsize=(5 * df["scenario"].nunique(), 4),
                             sharey=True)
    if df["scenario"].nunique() == 1:
        axes = [axes]
    for ax, (scen, gscen) in zip(axes, df.groupby("scenario")):
        for fname, gflt in gscen.groupby("filter"):
            curve = gflt.groupby("T_sup")["collided"].mean()
            ax.plot(curve.index, curve.values * 100,
                    marker="o", linewidth=2, label=fname)
        ax.set_xscale("log")
        ax.set_xlabel("Supervisor latency $T_{sup}$ (s)")
        ax.set_ylabel("Collision rate (%)")
        ax.set_title(f"Scenario {scen}")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.4)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                   label="_1% target")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                    fontsize=9, framealpha=0.95)
    fig.suptitle("Collision rate vs supervisor latency (SafeBench-LLM)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = out_dir / "fig_collision_vs_tsup.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_intervention_vs_safety(df, out_dir):
    """Pareto-style: intervention (x) vs collision rate (y), per filter."""
    fig, ax = plt.subplots(figsize=(7, 5))
    g = df.groupby("filter").agg(coll=("collided", "mean"),
                                  interv=("interv", "mean")).reset_index()
    for _, row in g.iterrows():
        ax.scatter(row["interv"], row["coll"] * 100, s=140,
                   edgecolor="k", linewidth=1.0, zorder=3)
        ax.annotate(row["filter"], (row["interv"], row["coll"] * 100),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9)
    ax.set_xlabel("Average intervention magnitude  ||u* - u_nom||")
    ax.set_ylabel("Collision rate (%)")
    ax.set_title("Safety vs intervention trade-off "
                 "(lower-left = better)")
    ax.grid(True, alpha=0.4)
    out = out_dir / "fig_pareto.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_h_min_heatmap(df, out_dir):
    """Heatmap: filter x T_sup, colored by min h over all seeds."""
    pivot = (df.groupby(["filter", "T_sup"])["h_min"]
               .min().unstack().fillna(0.0))
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto",
                   cmap="RdYlGn",
                   vmin=-50, vmax=50)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{t}s" for t in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Supervisor latency $T_{sup}$")
    ax.set_title("Worst-case barrier value $\\min_t h$ "
                 "(green = safe, red = violated)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:+.1f}",
                    ha="center", va="center",
                    color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="min h")
    fig.tight_layout()
    out = out_dir / "fig_hmin_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="/mnt/user-data/outputs/benchmark_results.csv")
    ap.add_argument("--out-dir",
                    default="/mnt/user-data/outputs/figures")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(fig_collision_vs_tsup(df, out_dir))
    paths.append(fig_intervention_vs_safety(df, out_dir))
    paths.append(fig_h_min_heatmap(df, out_dir))
    print("Wrote:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
