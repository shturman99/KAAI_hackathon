"""The same physical targets predicted from catalogue photometry alone.

The bar the embedding has to clear. Skill above this line is what the pixels
carry beyond aperture photometry; skill below it means the embedding is an
expensive way to recompute a colour. Which features count as "photometry"
is the photometry_baseline_features decision, and the headline claim of
goal 1 depends entirely on where that line is drawn.
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
    ap.add_argument("--cat", required=True)
    ap.add_argument("--eval-split", required=True)
    ap.add_argument("--real-quality-cuts", required=True)
    ap.add_argument("--target-transform", required=True)
    ap.add_argument("--probe-model", required=True)
    ap.add_argument("--probe-regularization", required=True)
    ap.add_argument("--photometry-baseline-features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
    X, feat_names = lc.photometry_features(cat, a.photometry_baseline_features)
    targets, names = lc.transform_targets(cat, a.target_transform)

    rows = []
    for t, y in targets.items():
        mask = lc.real_cut_mask(cat, a.real_quality_cuts, target_cols=[t])
        mask &= np.isfinite(y) & np.isfinite(X).all(axis=1)
        Xm, ym = X[mask], y[mask]
        folds = lc.make_folds(
            len(Xm), a.eval_split,
            ra=cat.loc[mask, "ra"].to_numpy() if "ra" in cat.columns else None,
            dec=cat.loc[mask, "dec"].to_numpy() if "dec" in cat.columns else None,
        )
        # Photometric features are on wildly different scales, so they are
        # always z-scored -- embedding_preprocessing is not in scope here.
        res = lc.evaluate_probe(
            Xm, ym, folds, "standardize",
            lambda: lc.make_regressor(a.probe_model, a.probe_regularization),
        )
        rows.append({
            "target": t,
            "fitted_quantity": names[t],
            "features": a.photometry_baseline_features,
            "feature_names": "|".join(feat_names),
            "n_features": X.shape[1],
            "n_rows": int(mask.sum()),
            "r2_mean": res["score_mean"],
            "r2_std": res["score_std"],
            "n_folds": res["n_folds"],
            "alpha_mean": res["alpha_mean"],
            "probe_model": a.probe_model,
            "eval_split": a.eval_split,
            "target_transform": a.target_transform,
            "real_quality_cuts": a.real_quality_cuts,
        })
        print(f"  {t:10s} R2 = {res['score_mean']:.4f} +/- {res['score_std']:.4f} "
              f"(n={mask.sum()}, {X.shape[1]} features)", file=sys.stderr)

    lc.write_table(a.out, pd.DataFrame(rows))


if __name__ == "__main__":
    sys.exit(main())
