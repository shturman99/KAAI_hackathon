"""A probe trained on real galaxies, applied unchanged to simulated ones.

A complementary read on the gap: a probe that transfers is reading something
both domains share, so transfer R2 says what survives the shift in a way an
AUC cannot. Scored against ASTRID's exact, particle-derived properties rather
than fitted ones.

Two comparisons travel with each number. The in-domain R2 is the same probe
scored on held-out *real* galaxies, so the drop on transfer is legible. The
sim-refit R2 is a probe trained and scored inside the simulated sample, which
separates "the embedding does not encode this for simulated galaxies" from
"the real-trained mapping does not carry over".
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
    ap.add_argument("--real", required=True)
    ap.add_argument("--real-cat", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--sim-cat", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--real-quality-cuts", required=True)
    ap.add_argument("--sim-quality-cuts", required=True)
    ap.add_argument("--eval-split", required=True)
    ap.add_argument("--probe-model", required=True)
    ap.add_argument("--probe-regularization", required=True)
    ap.add_argument("--target-transform", required=True)
    ap.add_argument("--sim-mass-column", required=True)
    ap.add_argument("--sim-sfr-column", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from sklearn.metrics import r2_score

    Xr = np.load(lc.resolve_input(a.real, "npy")).astype(np.float64)
    Xs = np.load(lc.resolve_input(a.sim, "npy")).astype(np.float64)
    rcat = pd.read_parquet(lc.resolve_input(a.real_cat, "parquet"))
    scat = pd.read_parquet(lc.resolve_input(a.sim_cat, "parquet"))

    rt, rnames = lc.transform_targets(rcat, a.target_transform)
    st, snames = lc.astrid_targets(scat, a.target_transform,
                                   a.sim_mass_column, a.sim_sfr_column)
    smask0 = lc.sim_cut_mask(scat, a.sim_quality_cuts)

    rows = []
    for t in lc.PHYSICAL_TARGETS:
        if t not in rt or t not in st:
            continue
        yr, ys = rt[t], st[t]
        rmask = lc.real_cut_mask(rcat, a.real_quality_cuts, target_cols=[t]) & np.isfinite(yr)
        smask = smask0 & np.isfinite(ys)
        Xrm, yrm = Xr[rmask], yr[rmask]
        Xsm, ysm = Xs[smask], ys[smask]

        # In-domain reference: the same protocol, scored on held-out real rows.
        folds = lc.make_folds(
            len(Xrm), a.eval_split,
            ra=rcat.loc[rmask, "ra"].to_numpy() if "ra" in rcat.columns else None,
            dec=rcat.loc[rmask, "dec"].to_numpy() if "dec" in rcat.columns else None,
        )
        indomain = lc.evaluate_probe(
            Xrm, yrm, folds, a.embedding_preprocessing,
            lambda: lc.make_regressor(a.probe_model, a.probe_regularization))

        # Transfer: fit on every real row, apply unchanged to the simulated
        # sample. The real sample sets the conditioning too -- re-fitting it on
        # the simulated rows would silently absorb part of the domain shift.
        model, prep = lc.fit_full(
            Xrm, yrm, a.embedding_preprocessing,
            lambda: lc.make_regressor(a.probe_model, a.probe_regularization))
        pred = model.predict(prep.transform(Xsm))
        transfer_r2 = float(r2_score(ysm, pred))

        # Sim-refit control: is the property there at all in the simulated
        # embeddings, independent of whether the real mapping carries over?
        sfolds = lc.make_folds(len(Xsm), a.eval_split)
        simrefit = lc.evaluate_probe(
            Xsm, ysm, sfolds, a.embedding_preprocessing,
            lambda: lc.make_regressor(a.probe_model, a.probe_regularization))

        rows.append({
            "target": t,
            "real_quantity": rnames[t],
            "sim_quantity": snames[t],
            "n_real": int(rmask.sum()),
            "n_sim": int(smask.sum()),
            "r2_in_domain_real": indomain["score_mean"],
            "r2_in_domain_real_std": indomain["score_std"],
            "r2_transfer_to_sim": transfer_r2,
            "r2_sim_refit": simrefit["score_mean"],
            "r2_sim_refit_std": simrefit["score_std"],
            "transfer_gap": indomain["score_mean"] - transfer_r2,
            "pred_mean_offset": float(np.mean(pred) - np.mean(ysm)),
            "pred_std_ratio": float(np.std(pred) / np.std(ysm)) if np.std(ysm) else np.nan,
            "target_transform": a.target_transform,
            "sim_mass_column": a.sim_mass_column,
            "sim_sfr_column": a.sim_sfr_column,
            "sim_quality_cuts": a.sim_quality_cuts,
            "probe_model": a.probe_model,
            "eval_split": a.eval_split,
        })
        print(f"  {t:10s} real {indomain['score_mean']:+.3f} -> sim "
              f"{transfer_r2:+.3f}  (sim-refit {simrefit['score_mean']:+.3f})",
              file=sys.stderr)

    lc.write_table(a.out, pd.DataFrame(rows))


if __name__ == "__main__":
    sys.exit(main())
