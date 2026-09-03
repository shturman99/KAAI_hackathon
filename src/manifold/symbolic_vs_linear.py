"""Held-out skill of the best symbolic expression against the linear probe.

The question is how much of the embedding-to-property relationship is
genuinely nonlinear rather than an artefact of the probe being linear. Both
sides are scored on the same protocol and the same target transform, so the
difference is attributable to the model class rather than to the setup.

The comparison is deliberately asymmetric and says so: the linear probe reads
all 1024 conditioned dimensions, while the symbolic expression reads only the
handful of rotation-quotiented coordinates it was given. A symbolic score
below the linear one is therefore not evidence against nonlinearity -- it is
the compression being paid for.
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
    ap.add_argument("--expr", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--cat", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--real-quality-cuts", required=True)
    ap.add_argument("--eval-split", required=True)
    ap.add_argument("--target-transform", required=True)
    ap.add_argument("--probe-model", required=True)
    ap.add_argument("--probe-regularization", required=True)
    ap.add_argument("--sr-validation", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    expr = pd.read_csv(lc.resolve_input(a.expr, "csv"))
    X = np.load(lc.resolve_input(a.emb, "npy")).astype(np.float64)
    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
    targets, tnames = lc.transform_targets(cat, a.target_transform)

    rows = []
    for t, y in targets.items():
        sub = expr[expr["target"] == t]
        if sub.empty:
            continue
        mask = lc.real_cut_mask(cat, a.real_quality_cuts, target_cols=[t]) & np.isfinite(y)
        folds = lc.make_folds(
            len(X[mask]), a.eval_split,
            ra=cat.loc[mask, "ra"].to_numpy() if "ra" in cat.columns else None,
            dec=cat.loc[mask, "dec"].to_numpy() if "dec" in cat.columns else None,
        )
        lin = lc.evaluate_probe(
            X[mask], y[mask], folds, a.embedding_preprocessing,
            lambda: lc.make_regressor(a.probe_model, a.probe_regularization))

        sr_r2 = sub["r2_holdout"].to_numpy(dtype=float)
        finite = np.isfinite(sr_r2)
        best = sub.loc[sub["r2_holdout"].idxmax()] if finite.any() else sub.iloc[0]
        rows.append({
            "target": t,
            "fitted_quantity": tnames[t],
            "r2_linear_probe": lin["score_mean"],
            "r2_linear_probe_std": lin["score_std"],
            "r2_symbolic_mean": float(sr_r2[finite].mean()) if finite.any() else np.nan,
            "r2_symbolic_std": float(sr_r2[finite].std(ddof=0)) if finite.any() else np.nan,
            "r2_symbolic_best_fold": float(best["r2_holdout"]),
            "best_expression": best["expression"],
            "best_complexity": int(best["complexity"]),
            "n_symbolic_folds": int(finite.sum()),
            "symbolic_minus_linear": (float(sr_r2[finite].mean()) - lin["score_mean"]
                                      if finite.any() else np.nan),
            "n_linear_features": int(X.shape[1]),
            "n_symbolic_features": int(best["sr_n_coordinates"]),
            "comparison_caveat":
                "the linear probe reads all conditioned embedding dimensions while "
                "the expression reads only the rotation-quotiented coordinates, so "
                "a deficit measures the cost of compression, not an absence of "
                "nonlinearity",
            "sr_validation": a.sr_validation,
            "probe_model": a.probe_model,
            "eval_split": a.eval_split,
            "target_transform": a.target_transform,
        })
        print(f"  {t:10s} linear {lin['score_mean']:+.3f}  symbolic "
              f"{rows[-1]['r2_symbolic_mean']:+.3f}", file=sys.stderr)

    lc.write_table(a.out, pd.DataFrame(rows))


if __name__ == "__main__":
    sys.exit(main())
