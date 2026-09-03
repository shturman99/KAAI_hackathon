"""The same morphology labels predicted from gzArm photometry alone.

gzArm ships fluxes rather than magnitudes and carries no sersic index, so
magnitudes are formed at the Legacy Survey zeropoint and any feature the
catalogue lacks is dropped -- the surviving feature list travels on every
row so the bar this sets is never ambiguous.
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
    ap.add_argument("--probe-model", required=True)
    ap.add_argument("--probe-regularization", required=True)
    ap.add_argument("--morphology-label-form", required=True)
    ap.add_argument("--photometry-baseline-features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
    X, feat_names = lc.photometry_features(cat, a.photometry_baseline_features)
    finite = np.isfinite(X).all(axis=1)

    rows = []
    for lab, y, ok, is_binary, thr in lc.morphology_targets(cat, a.morphology_label_form):
        m = ok & finite
        Xm, ym = X[m], y[m]
        task = "classification" if is_binary else "regression"
        folds = lc.make_folds(
            len(Xm), a.eval_split,
            ra=cat.loc[m, "ra"].to_numpy() if "ra" in cat.columns else None,
            dec=cat.loc[m, "dec"].to_numpy() if "dec" in cat.columns else None,
            stratify=ym if is_binary else None,
        )
        res = lc.evaluate_probe(
            Xm, ym, folds, "standardize",
            (lambda: lc.make_classifier(a.probe_model, a.probe_regularization))
            if is_binary else
            (lambda: lc.make_regressor(a.probe_model, a.probe_regularization)),
            task=task,
        )
        rows.append({
            "label": lab,
            "metric": "auc" if is_binary else "r2",
            "features": a.photometry_baseline_features,
            "feature_names": "|".join(feat_names),
            "n_features": X.shape[1],
            "n_rows": int(m.sum()),
            "positive_rate": float(np.mean(ym)) if is_binary else float("nan"),
            "threshold": thr,
            "score_mean": res["score_mean"],
            "score_std": res["score_std"],
            "n_folds": res["n_folds"],
            "morphology_label_form": a.morphology_label_form,
            "probe_model": a.probe_model,
            "eval_split": a.eval_split,
        })
        print(f"  {lab:24s} {'AUC' if is_binary else 'R2'} = "
              f"{res['score_mean']:.4f} +/- {res['score_std']:.4f} (n={m.sum()})",
              file=sys.stderr)

    lc.write_table(a.out, pd.DataFrame(rows))


if __name__ == "__main__":
    sys.exit(main())
