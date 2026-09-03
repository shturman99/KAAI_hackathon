"""Accuracy against the number of terms a symbolic expression is allowed.

One panel per target. The symbolic curve is the Pareto front scored on the
held-out fold; the two dashed references are ridge at matched budget and ridge
on the full embedding. Where the symbolic curve plateaus is the answer to
"how many terms does this relationship actually need"; the distance to the
linear ceiling is what compression costs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = lc.by_output_id(a.inputs)
    df = lc.load_csv(d["symbolic_complexity_curve"])
    plt = lc.figure_style()

    targets = [t for t in lc.PHYSICAL_TARGETS if t in set(df["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.1 * len(targets), 3.7),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, t in zip(axes, targets):
        sub = df[df["target"] == t]
        name = sub["fitted_quantity"].iloc[0]

        sym = sub[sub["model"] == "symbolic"]
        # best held-out R2 achievable at or below each complexity: the curve a
        # reader actually cares about is monotone-best-so-far, not the raw
        # scatter of every front point across folds.
        g = (sym.groupby(["fold", "complexity"])["r2_holdout"].max().reset_index())
        grid = np.arange(1, int(g["complexity"].max()) + 1) if len(g) else np.array([1])
        curves = []
        for f, gf in g.groupby("fold"):
            gf = gf.sort_values("complexity")
            best = [gf.loc[gf["complexity"] <= c, "r2_holdout"].max() for c in grid]
            curves.append(best)
        curves = np.array(curves, dtype=float)
        with np.errstate(invalid="ignore"):
            mu, sd = np.nanmean(curves, 0), np.nanstd(curves, 0)
        ax.plot(grid, mu, "-o", ms=3.2, lw=1.5, color="#1f4e79", label="symbolic")
        ax.fill_between(grid, mu - sd, mu + sd, color="#1f4e79", alpha=.18, lw=0)

        lk = sub[sub["model"] == "linear_k"].groupby("n_terms")["r2_holdout"].mean()
        if len(lk):
            ax.plot(lk.index, lk.values, "--s", ms=3, lw=1.2, color="#b06d1a",
                    label="ridge on k coords")
        la = sub[sub["model"] == "linear_all"]["r2_holdout"].mean()
        ax.axhline(la, ls=":", lw=1.4, color="#8c2f39",
                   label=f"ridge, all 1024 dims ({la:.2f})")

        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("expression complexity (nodes)")
        ax.set_xlim(0.5, min(30, grid.max()) + .5)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("held-out R²")
    axes[0].set_ylim(-0.05, 1.0)
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("How much accuracy does each extra term buy?  "
                 "Pareto front scored out-of-sample", fontsize=11, y=1.02)
    fig.tight_layout()
    lc.write_figure(a.out, fig)


if __name__ == "__main__":
    sys.exit(main())
