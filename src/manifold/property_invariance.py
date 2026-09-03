"""Do orientation-free properties stay still when the picture is turned?

Stellar mass, star-formation rate, metallicity, redshift and morphological
class are all properties of a galaxy, not of how it happens to be angled on
the sky. A probe reading them off the embedding should therefore return the
same answer for every rotation of the same galaxy. Whether it does is a
separate question from whether the embedding moves at all: the orbit can be
long and the readout still flat, if rotation moves the embedding in
directions the probe does not weight.

The measurement is the spread of a probe's prediction across one galaxy's
rotation orbit, and it is reported against two scales that give it meaning:

  * the probe's own out-of-fold residual RMSE -- rotation is a nuisance worth
    worrying about only if it moves the prediction by something comparable to
    the error the probe already makes;
  * the population spread of the target -- how much of the real astrophysical
    range the encoder manufactures out of nothing but orientation.

Morphology probes are fitted on gzArm and applied to the anchor orbits. Those
are different galaxies, so no accuracy is claimed here -- but stability under
rotation is a property of the fitted function, and can be measured anywhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc


def fit_and_scale(X, y, folds, prep_mode, make_model, task="regression"):
    """Fit on all rows, and return the model plus its honest error scale.

    The model applied to the orbits is fitted on the whole sample, because the
    orbit galaxies are the measurement set and no score is read off them. The
    error scale it is judged against is out-of-fold, so it is not optimistic.
    """
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        p = lc.EmbeddingPreprocessor(prep_mode).fit(X[tr])
        m = make_model()
        m.fit(p.transform(X[tr]), y[tr])
        Z = p.transform(X[te])
        oof[te] = (m.predict_proba(Z)[:, 1] if task == "classification"
                   and hasattr(m, "predict_proba") else m.predict(Z))
    prep = lc.EmbeddingPreprocessor(prep_mode).fit(X)
    model = make_model()
    model.fit(prep.transform(X), y)
    resid = float(np.sqrt(np.nanmean((oof - y) ** 2)))
    return model, prep, resid


def predict(model, prep, E, task):
    Z = prep.transform(E)
    if task == "classification" and hasattr(model, "predict_proba"):
        return model.predict_proba(Z)[:, 1]
    return model.predict(Z)


def orbit_stats(pred, g, n_ang):
    """Per-galaxy spread of a prediction across its own rotation orbit."""
    P = pred.reshape(g, n_ang)
    return {
        "orbit_std_median": float(np.median(P.std(axis=1, ddof=0))),
        "orbit_std_mean": float(P.std(axis=1, ddof=0).mean()),
        "orbit_ptp_median": float(np.median(P.max(axis=1) - P.min(axis=1))),
        "orbit_ptp_p95": float(np.percentile(P.max(axis=1) - P.min(axis=1), 95)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orbits", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--cat", required=True)
    ap.add_argument("--gz-emb", required=True)
    ap.add_argument("--gz-cat", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--real-quality-cuts", required=True)
    ap.add_argument("--eval-split", required=True)
    ap.add_argument("--target-transform", required=True)
    ap.add_argument("--probe-model", required=True)
    ap.add_argument("--probe-regularization", required=True)
    ap.add_argument("--morphology-label-form", required=True)
    # The morphology probes are L2 logistic with an inner 12-point penalty
    # grid over 5 folds -- 60 solves per fit at 1024 features, which is why
    # probes.morphology_auc costs ~30 min. The physical targets are ridge and
    # cost about a minute. Splitting them lets the cheap half be materialized
    # on its own; the artifact records which half it holds.
    ap.add_argument("--include", default="all",
                    choices=["all", "physical", "morphology"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    orbits = np.load(lc.resolve_input(a.orbits, "npy")).astype(np.float64)
    g, n_ang, d = orbits.shape
    flat = orbits.reshape(g * n_ang, d)

    X = np.load(lc.resolve_input(a.emb, "npy")).astype(np.float64)
    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
    rows = []

    # ---- physical properties, fitted on the anchor sample -------------------
    targets, names = lc.transform_targets(cat, a.target_transform)
    for t, y in (targets.items() if a.include in ('all','physical') else []):
        mask = lc.real_cut_mask(cat, a.real_quality_cuts, target_cols=[t])
        mask &= np.isfinite(y)
        Xm, ym = X[mask], y[mask]
        folds = lc.make_folds(
            len(Xm), a.eval_split,
            ra=cat.loc[mask, "ra"].to_numpy() if "ra" in cat.columns else None,
            dec=cat.loc[mask, "dec"].to_numpy() if "dec" in cat.columns else None,
        )
        model, prep, resid = fit_and_scale(
            Xm, ym, folds, a.embedding_preprocessing,
            lambda: lc.make_regressor(a.probe_model, a.probe_regularization))
        st = orbit_stats(predict(model, prep, flat, "regression"), g, n_ang)
        tgt_sd = float(np.std(ym, ddof=0))
        rows.append({
            "property": t, "fitted_quantity": names[t], "kind": "physical",
            "units": "target units (dex where logged)",
            "fitted_on": "anchor", "n_fit_rows": int(mask.sum()),
            **st,
            "probe_residual_rmse": resid,
            "target_population_std": tgt_sd,
            "orbit_std_over_residual": st["orbit_std_median"] / resid,
            "orbit_std_over_target_std": st["orbit_std_median"] / tgt_sd,
        })
        print(f"  {t:10s} orbit sd {st['orbit_std_median']:.4f} vs residual "
              f"{resid:.4f}  ->  {st['orbit_std_median'] / resid:.3f}",
              file=sys.stderr)

    # ---- morphology, fitted on gzArm, applied to the anchor orbits ----------
    gX = np.load(lc.resolve_input(a.gz_emb, "npy")).astype(np.float64)
    gcat = pd.read_parquet(lc.resolve_input(a.gz_cat, "parquet"))
    morph = (lc.morphology_targets(gcat, a.morphology_label_form)
             if a.include in ('all','morphology') else [])
    for lab, y, ok, is_bin, thr in morph:
        Xm, ym = gX[ok], y[ok]
        task = "classification" if is_bin else "regression"
        folds = lc.make_folds(len(Xm), a.eval_split,
                              stratify=ym if is_bin else None)
        make = ((lambda: lc.make_classifier(a.probe_model, a.probe_regularization))
                if is_bin else
                (lambda: lc.make_regressor(a.probe_model, a.probe_regularization)))
        model, prep, resid = fit_and_scale(
            Xm, ym, folds, a.embedding_preprocessing, make, task=task)
        st = orbit_stats(predict(model, prep, flat, task), g, n_ang)
        tgt_sd = float(np.std(ym, ddof=0))
        rows.append({
            "property": lab, "fitted_quantity": f"P({lab} > {thr:.4g})",
            "kind": "morphology",
            "units": "probability" if is_bin else "vote fraction",
            "fitted_on": "gzArm", "n_fit_rows": int(ok.sum()),
            **st,
            "probe_residual_rmse": resid,
            "target_population_std": tgt_sd,
            "orbit_std_over_residual": st["orbit_std_median"] / resid,
            "orbit_std_over_target_std": st["orbit_std_median"] / tgt_sd,
        })
        print(f"  {lab:24s} orbit sd {st['orbit_std_median']:.4f} vs residual "
              f"{resid:.4f}  ->  {st['orbit_std_median'] / resid:.3f}",
              file=sys.stderr)

    df = pd.DataFrame(rows)
    df["included"] = a.include
    df["n_galaxies"] = g
    df["n_angles"] = n_ang
    df["embedding_preprocessing"] = a.embedding_preprocessing
    df["eval_split"] = a.eval_split
    df["probe_model"] = a.probe_model
    df["morphology_label_form"] = a.morphology_label_form
    df["morphology_caveat"] = (
        "morphology probes are fitted on gzArm and applied to anchor orbits; "
        "these are different galaxies, so no accuracy is claimed -- only the "
        "stability of the fitted function under rotation")
    df["interpretation"] = (
        "orbit_std_over_residual is the headline: the spread a probe's "
        "prediction shows across one galaxy's rotation orbit, divided by the "
        "probe's own out-of-fold error. Well below 1 means the readout is "
        "effectively orientation-invariant; near or above 1 means turning the "
        "picture moves the answer as much as the probe's own error does.")
    lc.write_table(a.out, df)


if __name__ == "__main__":
    sys.exit(main())
