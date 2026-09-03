"""How much of the real-versus-simulated separation each control accounts for.

The left panel is the whole argument of goal 3 in one axis: the raw AUC is
near 1 and means almost nothing, and what matters is how far the controls
move it. The limitation is printed on the figure rather than left to the
caption, because the residual number is not interpretable without it.
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
    raw = lc.load_json(d["raw_discriminability"])
    matched = lc.load_json(d["matched_discriminability"])
    resid = lc.load_json(d["residual_discriminability"])
    transfer = lc.load_csv(d["probe_transfer_r2"])

    plt = lc.figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))

    # --- the three stages --------------------------------------------------
    ax = axes[0]
    stages = [("raw\n(no controls)", raw, "#8c2f39"),
              ("+ matched in\nmass & redshift", matched, "#b06d1a"),
              (f"+ instrument subspace\nprojected out "
               f"({resid.get('n_directions', 0)}d)", resid, "#1f4e79")]
    xs = np.arange(len(stages))
    vals = [s[1]["auc_mean"] for s in stages]
    errs = [s[1]["auc_std"] for s in stages]
    ax.bar(xs, vals, yerr=errs, color=[s[2] for s in stages], width=0.6,
           error_kw={"elinewidth": 1, "ecolor": "0.35", "capsize": 3})
    for x, v, e, s in zip(xs, vals, errs, stages):
        ax.text(x, v + e + 0.012, f"{v:.3f}\nn={s[1]['n_sim_used']}+{s[1]['n_real_used']}",
                ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0.5, color="0.5", ls=":", lw=1)
    ax.text(len(stages) - 0.45, 0.5, " chance", va="bottom", ha="right",
            fontsize=7.5, color="0.35")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in stages], fontsize=8)
    ax.set_ylabel("held-out AUC, real vs simulated")
    ax.set_ylim(0.45, 1.06)
    ax.set_title("Discriminability under successive controls")
    ax.grid(axis="x", visible=False)

    # --- probe transfer ----------------------------------------------------
    ax = axes[1]
    t = transfer.copy()
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["r2_in_domain_real"], height=0.36, color="#1f4e79",
            label="real, held out")
    ax.barh(y - 0.2, t["r2_transfer_to_sim"], height=0.36, color="#b06d1a",
            label="transferred to ASTRID")
    ax.plot(t["r2_sim_refit"], y - 0.2, "k|", ms=9, mew=1.4,
            label="refit inside ASTRID")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(t["real_quantity"], fontsize=8)
    ax.set_xlabel("R²")
    ax.set_title("Does a real-trained probe transfer?")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.grid(axis="y", visible=False)

    fig.suptitle("Real versus simulated: what survives the controls", fontsize=11, y=1.02)
    fig.text(0.5, -0.06, "Limitation: " + resid.get("limitation", ""),
             ha="center", va="top", fontsize=7, color="0.3", wrap=True)
    fig.tight_layout()
    lc.write_figure(a.out, fig)


if __name__ == "__main__":
    sys.exit(main())
