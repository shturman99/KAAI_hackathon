"""How well can a closed-form expression reproduce the relationship, as a
function of how many terms it is allowed?

`symbolic_expressions` reduces the Pareto front to a single expression by a
pre-registered rule. That rule answers "which one expression should we quote",
not "how much accuracy does each extra term buy" -- and the second question is
the one that says whether a compact symbolic model is viable at all.

This walks the whole front instead. For every complexity the search reached, it
refits the constants on the training fold and scores the expression on the
held-out fold, so each point on the curve is a genuine out-of-sample number
rather than the search's own training error.

Two references travel with it, both at matched budget:

  linear_k   ridge on the k leading quotient coordinates, k = 1..n. The bar a
             symbolic expression has to justify itself against: if a k-term
             expression cannot beat a k-coordinate linear fit, the symbolic
             form is buying interpretability and nothing else.
  linear_all ridge on the full conditioned embedding. The ceiling.

Complexity is node count, so `c * z2` is 3. The number of *distinct
coordinates* an expression reads is reported separately as `n_terms`, since
that is the more intuitive reading of "how many terms".
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gp_symbolic as gp
from symbolic_regression import N_COORDS, build_coordinates, run_search, Standardiser


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orbits", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--cat", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--real-quality-cuts", required=True)
    ap.add_argument("--eval-split", required=True)
    ap.add_argument("--target-transform", required=True)
    ap.add_argument("--quotient-operator", required=True)
    ap.add_argument("--sr-reduction", required=True)
    ap.add_argument("--sr-n-coordinates", required=True)
    ap.add_argument("--sr-operator-set", required=True)
    ap.add_argument("--sr-validation", required=True)
    ap.add_argument("--max-terms", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from sklearn.metrics import r2_score

    k = N_COORDS[a.sr_n_coordinates]
    orbits = np.load(lc.resolve_input(a.orbits, "npy")).astype(np.float64)
    anchor_raw = np.load(lc.resolve_input(a.emb, "npy")).astype(np.float64)
    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))

    prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing).fit(anchor_raw)
    anchor = prep.transform(anchor_raw)
    orb = prep.transform(orbits.reshape(-1, orbits.shape[2])).reshape(orbits.shape)
    Q, _ = lc.quotient_representation(orb, a.quotient_operator, full_cloud=anchor)
    C, same_space, rinfo = build_coordinates(Q, anchor, a.sr_reduction, k)

    targets, tnames = lc.transform_targets(cat, a.target_transform)
    row_ids = np.arange(len(anchor)) if same_space else np.load(
        Path(a.orbits) / "orbit_row_index.npy")
    base = lc.real_cut_mask(cat, a.real_quality_cuts)

    rows = []
    for t in lc.PHYSICAL_TARGETS:
        if t not in targets:
            continue
        y_all = targets[t]
        keep = base[row_ids] & np.isfinite(y_all[row_ids])
        Xc, yc = C[keep], y_all[row_ids][keep]
        Xe = anchor[row_ids][keep]                     # for the linear ceiling
        folds = lc.make_folds(len(Xc), a.eval_split)

        for fi, (tr, te) in enumerate(folds):
            sc = Standardiser().fit(Xc[tr], yc[tr])
            Xs_, ys_ = sc.X(Xc), sc.y(yc)
            seed = lc.SEED + 1000 * fi + zlib.crc32(t.encode()) % 997
            res = run_search(Xs_[tr], ys_[tr], a.sr_operator_set, k, seed)

            for p in res.pareto_:
                expr = (gp.refit_constants(p["expr"], Xs_[tr], ys_[tr])
                        if a.sr_validation != "holdout_only" else p["expr"])
                pred = gp.evaluate(expr, Xs_[te])
                r2 = float(r2_score(ys_[te], pred)) if np.all(np.isfinite(pred)) else np.nan
                _, mae = gp.SymbolicRegressor.losses(expr, Xs_[te], ys_[te])
                rows.append({
                    "target": t, "fitted_quantity": tnames[t], "fold": fi,
                    "model": "symbolic", "complexity": p["complexity"],
                    "n_terms": len(gp.variables(expr)),
                    "n_constants": len(gp.constants(expr)),
                    "r2_holdout": r2, "mae_holdout": mae,
                    "expression": gp.to_string(expr, [f"z{i}" for i in range(k)]),
                })

            # linear reference at matched budget, k coordinates at a time
            for kk in range(1, min(a.max_terms, k) + 1):
                r = lc.evaluate_probe(Xc[:, :kk], yc, [(tr, te)], "standardize",
                                      lambda: lc.SvdRidgeCV())
                rows.append({
                    "target": t, "fitted_quantity": tnames[t], "fold": fi,
                    "model": "linear_k", "complexity": np.nan, "n_terms": kk,
                    "n_constants": kk + 1, "r2_holdout": r["score_mean"],
                    "mae_holdout": np.nan, "expression": f"ridge on z0..z{kk-1}",
                })
            r = lc.evaluate_probe(Xe, yc, [(tr, te)], "standardize",
                                  lambda: lc.SvdRidgeCV())
            rows.append({
                "target": t, "fitted_quantity": tnames[t], "fold": fi,
                "model": "linear_all", "complexity": np.nan,
                "n_terms": int(Xe.shape[1]), "n_constants": int(Xe.shape[1]) + 1,
                "r2_holdout": r["score_mean"], "mae_holdout": np.nan,
                "expression": "ridge on the full conditioned embedding",
            })
            sym = [x["r2_holdout"] for x in rows
                   if x["target"] == t and x["fold"] == fi
                   and x["model"] == "symbolic" and np.isfinite(x["r2_holdout"])]
            best_r2 = max(sym) if sym else float("nan")
            print(f"  {t:10s} fold {fi}: front {len(res.pareto_):2d} pts, "
                  f"best held-out R2 {best_r2:+.3f}", file=sys.stderr)

    df = pd.DataFrame(rows)
    for key, val in rinfo.items():
        if not isinstance(val, (list, dict)):
            df[f"reduction_{key}"] = val
    df["sr_operator_set"] = a.sr_operator_set
    df["sr_reduction"] = a.sr_reduction
    df["quotient_operator"] = a.quotient_operator
    df["eval_split"] = a.eval_split
    df["target_transform"] = a.target_transform
    df["coordinate_basis_caveat"] = (
        "expressions are in standardised PCA coordinates of the quotiented "
        "cloud; a PCA basis is arbitrary up to rotation, so a term is not a "
        "named physical direction")
    lc.write_table(a.out, df)


if __name__ == "__main__":
    sys.exit(main())
