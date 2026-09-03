"""Closed-form expressions relating rotation-quotiented coordinates to physics.

Compression first, and not as an optimisation: symbolic search scales
exponentially with the number of input variables, so 1024 raw dimensions are
not merely expensive but intractable. The pipeline is therefore

  measured rotation orbits -> quotient operator -> reduction to k coordinates
  -> genetic search over a bounded operator set -> Pareto front -> one
  expression chosen by a rule fixed in advance -> held-out score.

Where the quotient lives in the same ambient space as the shipped embeddings
(orbit_mean, pca_deflation) the reduction is *fitted* on the quotiented cloud
and then *applied* to the full anchor cloud, so the coordinates are
rotation-derived while the regression still gets the whole sample to learn
from. Where it does not (harmonic_magnitudes changes the ambient dimension),
the fit is restricted to the orbit subsample and the artifact says so.

One caveat travels with every expression: a PCA basis is arbitrary up to
rotation, so the coordinates carry no intrinsic meaning and an expression in
them is a statement about the manifold, not about a named embedding
direction.
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

N_COORDS = {"k4": 4, "k8": 8, "k16": 16}
SEARCH_ROWS = 2000     # rows the search sees; scoring always uses the full fold


def build_coordinates(Q, anchor, reduction, k, targets_for_probe=None, seed=lc.SEED):
    """Fit the reduction on the quotiented cloud; return coordinates + info."""
    from sklearn.decomposition import PCA, SparsePCA

    Qs = lc.EmbeddingPreprocessor("standardize").fit(Q)
    Qz = Qs.transform(Q)
    same_space = anchor is not None and anchor.shape[1] == Q.shape[1]
    source = Qs.transform(anchor) if same_space else Qz

    if reduction == "pca":
        m = PCA(n_components=k, random_state=seed).fit(Qz)
        C = m.transform(source)
        info = {"sr_reduction": reduction,
                "explained_variance_ratio": [float(v) for v in m.explained_variance_ratio_],
                "cumulative_explained_variance": float(m.explained_variance_ratio_.sum())}
    elif reduction == "sparse_pca":
        m = SparsePCA(n_components=k, random_state=seed, max_iter=200).fit(Qz)
        C = m.transform(source)
        nz = (np.abs(m.components_) > 1e-8).sum(axis=1)
        info = {"sr_reduction": reduction,
                "loadings_per_component": [int(v) for v in nz]}
    elif reduction == "probe_orthogonal":
        # Coordinates built from the linear probe directions, then
        # orthogonalised. Aligns the basis with known physics, but makes any
        # comparison against the linear probe partly circular.
        if not targets_for_probe:
            raise SystemExit("probe_orthogonal needs targets to fit probe directions")
        W = []
        for y in targets_for_probe:
            ok = np.isfinite(y)
            r = lc.SvdRidgeCV().fit(Qz[ok], y[ok])
            W.append(r.coef_)
        W = np.vstack(W)
        if len(W) < k:
            # Not enough probes to span k directions: complete the basis with
            # principal directions of what the probes leave behind.
            resid = Qz - (Qz @ W.T) @ np.linalg.pinv(W).T
            extra = PCA(n_components=k - len(W), random_state=seed).fit(resid)
            W = np.vstack([W, extra.components_])
        Bq, _ = np.linalg.qr(W[:k].T)
        C = source @ Bq
        info = {"sr_reduction": reduction, "n_probe_directions": min(k, len(W)),
                "circularity_caveat":
                    "coordinates are built from linear probe directions, so a "
                    "comparison against the linear probe is partly circular"}
    else:
        raise ValueError(f"unknown sr_reduction: {reduction}")

    info["fitted_on_quotient_applied_to"] = "anchor cloud" if same_space else "orbit subsample"
    info["n_coordinates"] = int(k)
    return C, same_space, info


class Standardiser:
    """Centre and scale the search space.

    The genetic search draws its constants from N(0, 2), so it can only reach
    coefficients of that order. Raw PCA coordinates have scale ~8 and a target
    like logMstar sits near 10.8 with a spread of 0.5, which needs coefficients
    near 0.05 -- unreachable in practice, and the search collapses onto a bare
    constant. Standardising both sides puts the wanted coefficients in the same
    range as the constants the search actually samples.

    R-squared is invariant under a common affine map of truth and prediction,
    so a score computed in this space is the score in the original units. The
    expressions, however, are in *standardised* coordinates, and the artifact
    says so.
    """

    def fit(self, X, y):
        self.xm, self.xs = X.mean(0), X.std(0)
        self.xs[self.xs == 0] = 1.0
        self.ym = float(np.mean(y))
        self.ys = float(np.std(y)) or 1.0
        return self

    def X(self, X):
        return (X - self.xm) / self.xs

    def y(self, y):
        return (np.asarray(y, float) - self.ym) / self.ys

    def inv_y(self, z):
        return np.asarray(z, float) * self.ys + self.ym


def run_search(Xtr, ytr, ops, k, seed, generations=40, population=600):
    rng = np.random.default_rng(seed)
    if len(Xtr) > SEARCH_ROWS:
        sub = rng.choice(len(Xtr), SEARCH_ROWS, replace=False)
        Xs, ys = Xtr[sub], ytr[sub]
    else:
        Xs, ys = Xtr, ytr
    r = gp.SymbolicRegressor(operator_set=ops, n_features=k, seed=seed,
                             generations=generations, population=population)
    return r.fit(Xs, ys)


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
    ap.add_argument("--sr-complexity-selection", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from sklearn.metrics import r2_score

    if a.sr_n_coordinates not in N_COORDS:
        raise ValueError(f"unknown sr_n_coordinates: {a.sr_n_coordinates}")
    k = N_COORDS[a.sr_n_coordinates]

    orbits = np.load(lc.resolve_input(a.orbits, "npy")).astype(np.float64)
    orbit_rows = np.load(Path(a.orbits) / "orbit_row_index.npy")
    anchor_raw = np.load(lc.resolve_input(a.emb, "npy")).astype(np.float64)
    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))

    prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing).fit(anchor_raw)
    anchor = prep.transform(anchor_raw)
    orb = prep.transform(orbits.reshape(-1, orbits.shape[2])).reshape(orbits.shape)

    Q, qinfo = lc.quotient_representation(orb, a.quotient_operator, full_cloud=anchor)
    targets, tnames = lc.transform_targets(cat, a.target_transform)

    C, same_space, rinfo = build_coordinates(
        Q, anchor, a.sr_reduction, k,
        targets_for_probe=[targets[t][orbit_rows] for t in lc.PHYSICAL_TARGETS
                           if t in targets] if a.sr_reduction == "probe_orthogonal" else None)

    # Rows the regression is fitted on: the whole anchor sample when the
    # coordinates are defined there, the orbit subsample otherwise.
    row_ids = np.arange(len(anchor)) if same_space else orbit_rows
    base_mask = lc.real_cut_mask(cat, a.real_quality_cuts)

    ops = a.sr_operator_set
    if ops not in gp.OPERATOR_SETS:
        raise ValueError(f"unknown sr_operator_set: {ops}")
    coord_names = [f"z{i}" for i in range(k)]

    rows = []
    for t in lc.PHYSICAL_TARGETS:
        if t not in targets:
            continue
        y_all = targets[t]
        keep = base_mask[row_ids] & np.isfinite(y_all[row_ids])
        Xc, yc = C[keep], y_all[row_ids][keep]
        folds = lc.make_folds(len(Xc), a.eval_split)

        for fi, (tr, te) in enumerate(folds):
            # zlib.crc32, not hash(): Python salts string hashes per
            # process, which would make the search unreproducible.
            seed = lc.SEED + 1000 * fi + zlib.crc32(t.encode()) % 997
            # Scalers are fitted on the training rows only, so the held-out
            # fold never informs the space the search works in.
            sc = Standardiser().fit(Xc[tr], yc[tr])
            Xs_, ys_ = sc.X(Xc), sc.y(yc)

            if a.sr_validation == "nested_cv":
                # The search never sees the rows selection is judged on, and
                # selection never sees the rows the score is reported on.
                itr, iva = tr[: int(0.8 * len(tr))], tr[int(0.8 * len(tr)):]
                res = run_search(Xs_[itr], ys_[itr], ops, k, seed)
                front = []
                for p in res.pareto_:
                    mse, mae = gp.SymbolicRegressor.losses(p["expr"], Xs_[iva], ys_[iva])
                    front.append({**p, "mse": mse, "mae": mae})
                best, sinfo = gp.select_from_front(front, a.sr_complexity_selection)
                expr = gp.refit_constants(best["expr"], Xs_[tr], ys_[tr])
            else:
                res = run_search(Xs_[tr], ys_[tr], ops, k, seed)
                best, sinfo = gp.select_from_front(res.pareto_, a.sr_complexity_selection)
                expr = (gp.refit_constants(best["expr"], Xs_[tr], ys_[tr])
                        if a.sr_validation == "holdout_refit" else best["expr"])

            # Scored in the standardised space. R-squared is invariant under a
            # common affine map of truth and prediction, so this is the score
            # in the target's own units.
            pred = gp.evaluate(expr, Xs_[te])
            r2 = float(r2_score(ys_[te], pred)) if np.all(np.isfinite(pred)) else float("nan")
            _, mae = gp.SymbolicRegressor.losses(expr, Xs_[te], ys_[te])
            rows.append({
                "target": t,
                "fitted_quantity": tnames[t],
                "fold": fi,
                "expression": gp.to_string(expr, coord_names),
                "complexity": gp.complexity(expr),
                "r2_holdout": r2,
                "mae_holdout": mae,
                "mae_train": best["mae"],
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "front_size": len(res.pareto_),
                "front_complexities": "|".join(str(p["complexity"]) for p in res.pareto_),
                "sr_reduction": a.sr_reduction,
                "sr_n_coordinates": k,
                "sr_operator_set": ops,
                "sr_validation": a.sr_validation,
                "sr_complexity_selection": a.sr_complexity_selection,
                "selection_rule_detail": str(sinfo.get("rule", "")),
                "quotient_operator": a.quotient_operator,
                "coordinates_fitted_on": rinfo["fitted_on_quotient_applied_to"],
                "expressions_in_standardised_coordinates": True,
                "target_standardisation": "z-scored on the training fold; R2 is "
                                          "unaffected, expression constants are "
                                          "in standardised units",
                "coordinate_basis_caveat":
                    "a PCA basis is arbitrary up to rotation, so these coordinates "
                    "carry no intrinsic meaning and may need aligning before an "
                    "expression maps onto an interpretable quantity",
                "eval_split": a.eval_split,
                "target_transform": a.target_transform,
            })
            print(f"  {t:10s} fold {fi}: R2 = {r2:+.3f}  c={gp.complexity(expr):2d}  "
                  f"{gp.to_string(expr, coord_names)[:70]}", file=sys.stderr)

    df = pd.DataFrame(rows)
    for key, val in rinfo.items():
        if not isinstance(val, (list, dict)):
            df[f"reduction_{key}"] = val
    df["reduction_cumulative_explained_variance"] = rinfo.get(
        "cumulative_explained_variance", np.nan)
    lc.write_table(a.out, df)


if __name__ == "__main__":
    sys.exit(main())
