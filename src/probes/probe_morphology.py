"""Held-out AUC for Galaxy Zoo morphology labels from the gzArm embedding.

Morphology comes exclusively from gzArm: the anchor sample's morphology
columns are 1.9-3.2% covered. The two catalogues are different galaxies and
are never merged.
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
    ap.add_argument("--probe-model", required=True)
    ap.add_argument("--probe-regularization", required=True)
    ap.add_argument("--morphology-label-form", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    X = np.load(lc.resolve_input(a.emb, "npy")).astype(np.float64)
    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
    if len(cat) != len(X):
        raise ValueError(f"row misalignment: {len(X)} embeddings vs {len(cat)} catalogue rows")

    rows = []
    for lab, y, ok, is_binary, thr in lc.morphology_targets(cat, a.morphology_label_form):
        Xm, ym = X[ok], y[ok]
        task = "classification" if is_binary else "regression"
        folds = lc.make_folds(
            len(Xm), a.eval_split,
            ra=cat.loc[ok, "ra"].to_numpy() if "ra" in cat.columns else None,
            dec=cat.loc[ok, "dec"].to_numpy() if "dec" in cat.columns else None,
            stratify=ym if is_binary else None,
        )
        res = lc.evaluate_probe(
            Xm, ym, folds, a.embedding_preprocessing,
            (lambda: lc.make_classifier(a.probe_model, a.probe_regularization))
            if is_binary else
            (lambda: lc.make_regressor(a.probe_model, a.probe_regularization)),
            task=task,
        )
        rows.append({
            "label": lab,
            "metric": "auc" if is_binary else "r2",
            "features": "embedding",
            "n_features": X.shape[1],
            "n_rows": int(ok.sum()),
            "positive_rate": float(np.mean(ym)) if is_binary else float("nan"),
            "threshold": thr,
            "score_mean": res["score_mean"],
            "score_std": res["score_std"],
            "n_folds": res["n_folds"],
            "morphology_label_form": a.morphology_label_form,
            "probe_model": a.probe_model,
            "eval_split": a.eval_split,
            "embedding_preprocessing": a.embedding_preprocessing,
        })
        print(f"  {lab:24s} {'AUC' if is_binary else 'R2'} = "
              f"{res['score_mean']:.4f} +/- {res['score_std']:.4f} (n={ok.sum()})",
              file=sys.stderr)

    lc.write_table(a.out, pd.DataFrame(rows))


if __name__ == "__main__":
    sys.exit(main())
