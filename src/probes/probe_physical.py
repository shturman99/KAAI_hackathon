"""Held-out R2 for each core physical target predicted from the anchor embedding.

The baseline measurement of goal 1: what is linearly decodable from a frozen
image-only embedding at all. The number is only interpretable against the
photometry-only bar produced by probe_photometry.py, never against zero.
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
    ap.add_argument("--emb", required=True)
    ap.add_argument("--cat", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--eval-split", required=True)
    ap.add_argument("--real-quality-cuts", required=True)
    ap.add_argument("--target-transform", required=True)
    ap.add_argument("--probe-model", required=True)
    ap.add_argument("--probe-regularization", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    X = np.load(lc.resolve_input(a.emb, "npy")).astype(np.float64)
    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
    if len(cat) != len(X):
        raise ValueError(f"row misalignment: {len(X)} embeddings vs {len(cat)} catalogue rows")

    targets, names = lc.transform_targets(cat, a.target_transform)

    rows = []
    for t, y in targets.items():
        # The cut is applied per target so that a target with a few
        # non-finite rows does not shrink the sample for the others.
        mask = lc.real_cut_mask(cat, a.real_quality_cuts, target_cols=[t])
        mask &= np.isfinite(y)
        Xm, ym = X[mask], y[mask]
        folds = lc.make_folds(
            len(Xm), a.eval_split,
            ra=cat.loc[mask, "ra"].to_numpy() if "ra" in cat.columns else None,
            dec=cat.loc[mask, "dec"].to_numpy() if "dec" in cat.columns else None,
        )
        res = lc.evaluate_probe(
            Xm, ym, folds, a.embedding_preprocessing,
            lambda: lc.make_regressor(a.probe_model, a.probe_regularization),
        )
        rows.append({
            "target": t,
            "fitted_quantity": names[t],
            "features": "embedding",
            "n_features": X.shape[1],
            "n_rows": int(mask.sum()),
            "r2_mean": res["score_mean"],
            "r2_std": res["score_std"],
            "n_folds": res["n_folds"],
            "alpha_mean": res["alpha_mean"],
            "probe_model": a.probe_model,
            "eval_split": a.eval_split,
            "target_transform": a.target_transform,
            "embedding_preprocessing": a.embedding_preprocessing,
            "real_quality_cuts": a.real_quality_cuts,
        })
        print(f"  {t:10s} R2 = {res['score_mean']:.4f} +/- {res['score_std']:.4f} "
              f"(n={mask.sum()})", file=sys.stderr)

    lc.write_table(a.out, pd.DataFrame(rows))


if __name__ == "__main__":
    sys.exit(main())
