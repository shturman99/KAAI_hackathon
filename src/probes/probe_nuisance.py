"""Held-out R2 for observing conditions, plus the fitted probe directions.

High R2 here is a warning, not a result: it means the embedding carries
telescope rather than galaxy. The weight vectors this writes are the wire to
goal 3 -- they span the instrument subspace that
domain_gap.residual_discriminability projects out -- so the artifact carries
the directions themselves, not only the R2 values.

Each direction is written as a linear functional on the *raw* embedding, so a
consumer conditioning the embedding differently can still re-express it
(see EmbeddingPreprocessor.direction_to_transformed). Directions are unit
normalised, which makes every nuisance variable count equally when the
subspace is formed, with the pre-normalisation norm kept alongside.
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
    ap.add_argument("--probe-model", required=True)
    ap.add_argument("--probe-regularization", required=True)
    ap.add_argument("--nuisance-feature-set", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    X = np.load(lc.resolve_input(a.emb, "npy")).astype(np.float64)
    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
    if len(cat) != len(X):
        raise ValueError(f"row misalignment: {len(X)} embeddings vs {len(cat)} catalogue rows")

    Y, var_names = lc.nuisance_features(cat, a.nuisance_feature_set)
    base_mask = lc.real_cut_mask(cat, a.real_quality_cuts)

    rows = []
    for j, var in enumerate(var_names):
        y = Y[:, j]
        mask = base_mask & np.isfinite(y)
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
        # The direction is refit on every retained row: it is a description of
        # the subspace, not a held-out measurement, so it should use all data.
        model, prep = lc.fit_full(
            Xm, ym, a.embedding_preprocessing,
            lambda: lc.make_regressor(a.probe_model, a.probe_regularization),
        )
        w = lc.linear_weights(model, X.shape[1])
        if w is None:
            raise SystemExit(
                f"probe_model '{a.probe_model}' exposes no weight vector; "
                "domain_gap.residual_discriminability needs one. Use ridge or "
                "linear_ols, or switch nuisance_projection to cca_subspace."
            )
        v = w / prep.scale_          # back to a functional on the raw embedding
        norm = float(np.linalg.norm(v))
        v_unit = v / norm if norm > 0 else v

        row = {
            "variable": var,
            "r2_mean": res["score_mean"],
            "r2_std": res["score_std"],
            "n_folds": res["n_folds"],
            "n_rows": int(mask.sum()),
            "alpha_mean": res["alpha_mean"],
            "weight_norm": norm,
            "nuisance_feature_set": a.nuisance_feature_set,
            "embedding_preprocessing": a.embedding_preprocessing,
            "probe_model": a.probe_model,
            "eval_split": a.eval_split,
        }
        row.update({f"w{i:04d}": float(v_unit[i]) for i in range(v_unit.size)})
        rows.append(row)
        print(f"  {var:16s} R2 = {res['score_mean']:.4f} +/- {res['score_std']:.4f}",
              file=sys.stderr)

    lc.write_table(a.out, pd.DataFrame(rows))


if __name__ == "__main__":
    sys.exit(main())
