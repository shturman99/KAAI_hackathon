"""The dimensionality budget, with the evidence each number rests on.

Three bars are cheap to draw and easy to over-read, so the panels beside them
carry the diagnostics: the block-analysis curves that show whether each
estimate sits on a plateau or is still moving with sample size, and the
harmonic spectrum that says what shape the orbit actually has. A reader
should be able to see whether the numbers reflect a real plateau or an
arbitrary cut.
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
    full = lc.load_json(d["intrinsic_dim_full"])
    orbit = lc.load_json(d["intrinsic_dim_orbit"])
    quot = lc.load_json(d["intrinsic_dim_quotient"])
    spec = lc.load_csv(d["orbit_fourier_spectrum"])

    plt = lc.figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

    # --- the budget --------------------------------------------------------
    ax = axes[0]
    items = [("full cloud\n(as shipped)", full, "#1f4e79"),
             ("one rotation\norbit", orbit, "#2e7d32"),
             ("after quotienting\nrotation", quot, "#8c2f39")]
    xs = np.arange(len(items))
    vals = [it[1]["id"] for it in items]
    bars = ax.bar(xs, vals, color=[it[2] for it in items], width=0.6)
    for x, (label, js, _), v, b in zip(xs, items, vals, bars):
        if js.get("sampling_assumption_violated"):
            # Drawing this bar as if it were an estimate would be the single
            # most misleading thing on the figure, so it is hatched and the
            # assumption-free reading is printed in its place.
            b.set_hatch("////"); b.set_alpha(0.45)
            pr = js.get("linear_participation_ratio_mean")
            tag = (f"{v:.1f}\nNOT an estimate\n(uniform angle grid)"
                   + (f"\nlinear PR = {pr:.2f}" if pr is not None else ""))
        else:
            tag = f"{v:.1f}" + ("\n(lower bound)" if js.get("is_lower_bound") else "")
        ax.text(x, v, tag, ha="center", va="bottom", fontsize=7.5)
    ax.axhline(lc.ID_UNDERESTIMATION_THRESHOLD, color="0.5", ls=":", lw=1)
    ax.text(len(items) - 0.5, lc.ID_UNDERESTIMATION_THRESHOLD, " TwoNN becomes a\n lower bound",
            va="bottom", ha="right", fontsize=7, color="0.35")
    ax.set_xticks(xs)
    ax.set_xticklabels([it[0] for it in items], fontsize=8)
    ax.set_ylabel("intrinsic dimension")
    ax.set_title(f"Dimensionality budget ({full.get('id_estimator', '')})")
    ax.set_ylim(0, max(vals + [lc.ID_UNDERESTIMATION_THRESHOLD]) * 1.28)
    ax.grid(axis="x", visible=False)

    # --- the stability evidence -------------------------------------------
    ax = axes[1]
    any_curve = False
    for label, js, colour in items:
        curve = (js.get("diagnostic") or {}).get("curve") or []
        if not curve:
            continue
        any_curve = True
        n = [c["block_size"] for c in curve]
        v = [c["id_mean"] for c in curve]
        e = [c["id_std"] for c in curve]
        ax.errorbar(n, v, yerr=e, marker="o", ms=4, lw=1.2, capsize=2,
                    color=colour, label=label.replace("\n", " "))
    ax.set_xscale("log")
    ax.set_xlabel("points per block")
    ax.set_ylabel("estimated ID")
    ax.set_title("Block analysis:\ndoes the estimate move with N?", fontsize=9)
    if any_curve:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "no block curve recorded", ha="center", va="center",
                transform=ax.transAxes, color="0.4")

    # --- what the orbit looks like ----------------------------------------
    ax = axes[2]
    s = spec[spec["harmonic"] > 0]
    ax.bar(s["harmonic"], s["frac_of_varying_power_mean"],
           yerr=s["frac_of_varying_power_std"], color="#2e7d32", width=0.7,
           error_kw={"elinewidth": 0.8, "ecolor": "0.4"})
    ax.set_xlabel("angular harmonic")
    ax.set_ylabel("fraction of varying power")
    inv = float(spec["invariant_power_frac_mean"].iloc[0])
    ax.set_title(f"Orbit harmonic content\n(harmonic 0 holds {inv:.3f} of total power)",
                 fontsize=9)
    ax.set_xticks(s["harmonic"].astype(int))
    ax.grid(axis="x", visible=False)

    fig.suptitle("How much of the embedding cloud's dimensionality is orientation",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    lc.write_figure(a.out, fig)


if __name__ == "__main__":
    sys.exit(main())
