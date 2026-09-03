"""Embedding skill against the photometry-only bar, for every target.

The point of the figure is the diagonal: a target above it is one where the
pixels carry information catalogue photometry does not. The nuisance probes
are drawn separately and on their own terms, because a high value there is a
warning rather than an achievement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = lc.by_output_id(a.inputs)
    phys = lc.load_csv(d["probe_r2_physical"])
    photo = lc.load_csv(d["probe_r2_photometry_baseline"])
    morph = lc.load_csv(d["morphology_auc"])
    morphp = lc.load_csv(d["morphology_auc_photometry"])
    nuis = lc.load_csv(d["nuisance_probe_r2"])

    plt = lc.figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

    # --- physical targets: embedding vs photometry -------------------------
    ax = axes[0]
    m = phys.merge(photo, on="target", suffixes=("_emb", "_phot"))
    ax.plot([-0.05, 1], [-0.05, 1], color="0.6", lw=0.8, ls="--", zorder=1)
    ax.errorbar(m["r2_mean_phot"], m["r2_mean_emb"],
                xerr=m["r2_std_phot"], yerr=m["r2_std_emb"],
                fmt="o", ms=5, color="#1f4e79", ecolor="#8aa8c8",
                elinewidth=1, capsize=2, zorder=3)
    for _, r in m.iterrows():
        ax.annotate(r["fitted_quantity_emb"], (r["r2_mean_phot"], r["r2_mean_emb"]),
                    textcoords="offset points", xytext=(6, -2), fontsize=7.5)
    ax.set_xlabel("photometry-only R²")
    ax.set_ylabel("embedding R²")
    ax.set_title("Physical properties")
    ax.set_xlim(-0.05, 1); ax.set_ylim(-0.05, 1)

    # --- morphology: embedding vs photometry -------------------------------
    ax = axes[1]
    mm = morph.merge(morphp, on="label", suffixes=("_emb", "_phot"))
    lo = 0.45 if (mm["metric_emb"] == "auc").all() else -0.05
    ax.plot([lo, 1], [lo, 1], color="0.6", lw=0.8, ls="--", zorder=1)
    ax.errorbar(mm["score_mean_phot"], mm["score_mean_emb"],
                xerr=mm["score_std_phot"], yerr=mm["score_std_emb"],
                fmt="s", ms=5, color="#7b3f00", ecolor="#d0a878",
                elinewidth=1, capsize=2, zorder=3)
    for _, r in mm.iterrows():
        ax.annotate(r["label"].replace("Fraction", ""),
                    (r["score_mean_phot"], r["score_mean_emb"]),
                    textcoords="offset points", xytext=(6, -2), fontsize=7.5)
    metric = mm["metric_emb"].iloc[0].upper() if len(mm) else "AUC"
    ax.set_xlabel(f"photometry-only {metric}")
    ax.set_ylabel(f"embedding {metric}")
    ax.set_title("Morphology (gzArm)")
    ax.set_xlim(lo, 1); ax.set_ylim(lo, 1)

    # --- nuisance probes ---------------------------------------------------
    ax = axes[2]
    n = nuis.sort_values("r2_mean")
    ypos = np.arange(len(n))
    ax.barh(ypos, n["r2_mean"], xerr=n["r2_std"], color="#8c2f39",
            error_kw={"elinewidth": 1, "ecolor": "0.4"}, height=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(n["variable"], fontsize=7)
    ax.set_xlabel("R² of observing condition from embedding")
    ax.set_title("Instrument signature\n(high is a warning, not a result)",
                 fontsize=9)
    ax.grid(axis="y", visible=False)

    fig.suptitle("What the frozen embedding encodes, measured against "
                 "the photometry bar", fontsize=11, y=1.01)
    fig.tight_layout()
    lc.write_figure(a.out, fig)


if __name__ == "__main__":
    sys.exit(main())
