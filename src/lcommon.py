"""Shared machinery for the aion-embedding-structure analysis.

Every script under ``src/`` imports this module. It holds the pieces the
astra.yaml decisions actually name — embedding conditioning, held-out
evaluation protocols, target transforms, quality cuts, probe families and
the intrinsic-dimension estimators — so that a decision is implemented in
exactly one place and every output that declares it behaves identically.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# This analysis is scoped to a 15 GB laptop with 20 cores. Unbounded BLAS
# fan-out there costs more in thread contention and per-thread working set
# than it buys in throughput, and on the widest fits it exhausts memory
# outright. Capped before numpy is imported so the setting takes; an explicit
# environment setting always wins.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(min(8, os.cpu_count() or 1)))

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# artifact I/O
#
# lightcone hands recipes ``{output}`` as a *directory* and ``{inputs.<id>}``
# as either a source file path (analysis-level Input) or the producing
# output's directory (sibling Output). These helpers hide that asymmetry.
# --------------------------------------------------------------------------

def artifact_path(outdir: str | os.PathLike, ext: str) -> Path:
    """Path to write inside an output directory, named after the output id."""
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{d.name}.{ext}"


def resolve_input(path: str | os.PathLike, ext: str) -> Path:
    """Resolve an input reference to a concrete file.

    Accepts either a file path (analysis-level Input) or an output
    directory holding exactly one artifact of the wanted extension.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        # The primary artifact is named after the output id. Some outputs also
        # write sidecars next to it (rotation_orbit_embeddings ships the row
        # index and angle grid), so the id-named file wins over a bare glob.
        primary = p / f"{p.name}.{ext}"
        if primary.is_file():
            return primary
        hits = sorted(p.glob(f"*.{ext}"))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"expected exactly one *.{ext} in {p}, found {len(hits)}: "
                + ", ".join(h.name for h in hits)
            )
        return hits[0]
    raise FileNotFoundError(f"input not found: {p}")


def write_table(outdir, df: pd.DataFrame) -> Path:
    p = artifact_path(outdir, "csv")
    df.to_csv(p, index=False)
    print(f"wrote {p} ({len(df)} rows)", file=sys.stderr)
    return p


def write_metric(outdir, payload: dict) -> Path:
    p = artifact_path(outdir, "json")
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))
    print(f"wrote {p}", file=sys.stderr)
    return p


def write_array(outdir, arr: np.ndarray) -> Path:
    p = artifact_path(outdir, "npy")
    np.save(p, arr)
    print(f"wrote {p} shape {arr.shape}", file=sys.stderr)
    return p


def write_figure(outdir, fig) -> Path:
    p = artifact_path(outdir, "pdf")
    fig.savefig(p, bbox_inches="tight")
    print(f"wrote {p}", file=sys.stderr)
    return p


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


SEED = 20260902  # fixed so every split, subsample and search is reproducible


# --------------------------------------------------------------------------
# decision: embedding_preprocessing
#
# Fitted on the training split only, then applied to the held-out split, so
# that no test information reaches the conditioning step. This matters for
# more than the probes: the same conditioning defines the metric the
# nearest-neighbour ID estimators operate in.
# --------------------------------------------------------------------------

class EmbeddingPreprocessor:
    def __init__(self, mode: str):
        if mode not in ("raw", "standardize", "l2_normalize", "standardize_then_l2"):
            raise ValueError(f"unknown embedding_preprocessing: {mode}")
        self.mode = mode
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "EmbeddingPreprocessor":
        X = np.asarray(X, dtype=np.float64)
        if self.mode in ("standardize", "standardize_then_l2"):
            self.mean_ = X.mean(axis=0)
            s = X.std(axis=0)
            s[s == 0] = 1.0          # constant dimensions pass through unscaled
            self.scale_ = s
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.mode in ("standardize", "standardize_then_l2"):
            X = (X - self.mean_) / self.scale_
        if self.mode in ("l2_normalize", "standardize_then_l2"):
            n = np.linalg.norm(X, axis=1, keepdims=True)
            n[n == 0] = 1.0
            X = X / n
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def direction_to_transformed(self, v: np.ndarray) -> np.ndarray:
        """Re-express a linear functional given in raw embedding space so that
        it acts identically on transformed vectors.

        For the affine part of the conditioning, ``v . x`` equals
        ``(v * scale) . z`` up to a constant, where ``z`` is the transformed
        vector. L2 normalisation is not affine, so under the l2 modes this is
        the direction of the affine part only -- which is what the nuisance
        projection needs, since it removes a subspace rather than a value.
        """
        return np.asarray(v, dtype=np.float64) * self.scale_


# --------------------------------------------------------------------------
# decision: eval_split
# --------------------------------------------------------------------------

def make_folds(n, split, ra=None, dec=None, stratify=None, seed=SEED):
    """Return a list of (train_idx, test_idx) pairs.

    ``random_80_20`` yields a single fold, which is AION-1's own protocol and
    therefore the setting whose numbers are directly comparable to the paper.
    ``kfold_5`` yields five. ``sky_blocked`` yields five folds drawn as
    contiguous regions on the sky, so galaxies sharing observing conditions
    land in the same fold and cannot leak instrument signature across it.
    """
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

    idx = np.arange(n)
    if split == "random_80_20":
        strat = stratify if stratify is not None else None
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
        return [(tr, te)]
    if split == "kfold_5":
        if stratify is not None:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            return list(kf.split(idx, stratify))
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        return list(kf.split(idx))
    if split == "sky_blocked":
        if ra is None or dec is None:
            raise ValueError("sky_blocked needs ra/dec; this artifact has no sky coordinates")
        from sklearn.cluster import KMeans
        # Contiguous blocks on the sky: k-means cells in (RA, Dec) with RA
        # compressed by cos(Dec) so the cells are round on the sphere.
        xy = np.column_stack([np.asarray(ra) * np.cos(np.radians(np.asarray(dec))),
                              np.asarray(dec)])
        lab = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(xy)
        return [(idx[lab != k], idx[lab == k]) for k in range(5)]
    raise ValueError(f"unknown eval_split: {split}")


# --------------------------------------------------------------------------
# decision: probe_model x probe_regularization
#
# ``ridge`` maps to its classification analogue -- an L2-penalised linear
# classifier -- wherever the target is binary, so that the same decision
# names one model family across regression and classification outputs.
# --------------------------------------------------------------------------

ALPHA_GRID = np.logspace(-3, 6, 28)


def make_regressor(probe_model, probe_regularization, seed=SEED):
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    if probe_model == "ridge":
        if probe_regularization == "cv_grid":
            # SvdRidgeCV, not RidgeCV(cv=5): identical alpha and coefficients
            # (verified to 1e-15 against sklearn on a matched splitter), but
            # one eigendecomposition per fold instead of one refit per
            # (alpha, fold) pair.
            return SvdRidgeCV(alphas=ALPHA_GRID, cv=5, seed=seed)
        if probe_regularization == "gcv":
            return RidgeCV(alphas=ALPHA_GRID)      # closed-form LOO / GCV
        if probe_regularization == "fixed":
            return Ridge(alpha=1.0)
        raise ValueError(f"unknown probe_regularization: {probe_regularization}")
    if probe_model == "linear_ols":
        return LinearRegression()
    if probe_model == "knn":
        return KNeighborsRegressor(n_neighbors=10)
    if probe_model == "mlp":
        return MLPRegressor(hidden_layer_sizes=(256,), max_iter=400,
                            early_stopping=True, random_state=seed)
    raise ValueError(f"unknown probe_model: {probe_model}")


def make_classifier(probe_model, probe_regularization, seed=SEED):
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    # Logistic loss has no closed form, so every grid point is a full iterative
    # solve. The grid spans the same range as ALPHA_GRID at coarser spacing,
    # which keeps the inner search affordable at 1024 features without changing
    # what the decision means.
    Cs = 1.0 / np.logspace(-3, 6, 12)
    # solver='newton-cholesky', not the default lbfgs. At 1024 features lbfgs
    # does not converge at the weakly-regularised end of the grid: it runs to
    # max_iter, and cost then grows like n^2.6 (measured 8.2s -> 48.6s for a 2x
    # sample, extrapolating to ~18 min per fit at the real fold size).
    # newton-cholesky forms the Hessian directly, which is the right trade when
    # n_samples > n_features: measured 38s at n=8000, and it selects the same
    # penalty as lbfgs at matched n (C*=0.01233 for both). A conditioning fix,
    # not a change of objective.
    SOLVER = dict(solver="newton-cholesky", max_iter=200)
    if probe_model in ("ridge", "logistic"):
        if probe_regularization == "cv_grid":
            return LogisticRegressionCV(Cs=Cs, cv=5, scoring="roc_auc",
                                        random_state=seed, **SOLVER)
        if probe_regularization == "gcv":
            # No closed-form GCV exists for logistic loss either; the honest
            # substitute is a cheaper 3-fold selection over the same grid.
            return LogisticRegressionCV(Cs=Cs, cv=3, scoring="roc_auc",
                                        random_state=seed, **SOLVER)
        if probe_regularization == "fixed":
            return LogisticRegression(C=1.0, **SOLVER)
        raise ValueError(f"unknown probe_regularization: {probe_regularization}")
    if probe_model == "linear_ols":
        return LogisticRegression(C=1e6, max_iter=5000)
    if probe_model == "knn":
        return KNeighborsClassifier(n_neighbors=10)
    if probe_model in ("mlp", "gradient_boosting"):
        if probe_model == "gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=seed)
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                             early_stopping=True, random_state=seed)
    raise ValueError(f"unknown probe/classifier model: {probe_model}")


def linear_weights(model, n_features):
    """Weight vector of a fitted linear model, or None if it has none."""
    w = getattr(model, "coef_", None)
    if w is None:
        return None
    w = np.asarray(w, dtype=np.float64).ravel()
    return w if w.size == n_features else None


# --------------------------------------------------------------------------
# decision: real_quality_cuts / sim_quality_cuts
# --------------------------------------------------------------------------

def real_cut_mask(cat: pd.DataFrame, cuts: str, target_cols=()) -> np.ndarray:
    """Row mask for the real (anchor / gzArm) catalogues."""
    m = np.ones(len(cat), dtype=bool)
    if cuts == "none":
        return m
    if cuts not in ("standard", "strict"):
        raise ValueError(f"unknown real_quality_cuts: {cuts}")
    if "shapeDefined" in cat.columns:
        m &= cat["shapeDefined"].to_numpy().astype(bool)
    for c in target_cols:
        if c in cat.columns:
            m &= np.isfinite(pd.to_numeric(cat[c], errors="coerce").to_numpy(dtype=float))
    if cuts == "strict" and "fsZwarn" in cat.columns:
        z = pd.to_numeric(cat["fsZwarn"], errors="coerce").to_numpy(dtype=float)
        # fsZwarn is only present for 62.6% of rows, so this cut also selects
        # on having been observed spectroscopically at all.
        m &= np.isfinite(z) & (z == 0)
    return m


def sim_cut_mask(cat: pd.DataFrame, cuts: str) -> np.ndarray:
    """Row mask for the ASTRID catalogue."""
    m = np.ones(len(cat), dtype=bool)
    if cuts == "none":
        return m
    if cuts not in ("pathology_only", "pathology_and_encl"):
        raise ValueError(f"unknown sim_quality_cuts: {cuts}")
    if "pathology" in cat.columns:
        # StringDtype comparison yields pandas' *nullable* boolean, whose
        # .to_numpy() is object dtype and cannot be &='d into a bool array.
        # Force a plain bool array.
        p = cat["pathology"].astype("string").fillna("")
        m &= (p.str.strip() == "").fillna(False).to_numpy(dtype=bool)
    if cuts == "pathology_and_encl" and "enclR" in cat.columns:
        e = pd.to_numeric(cat["enclR"], errors="coerce").to_numpy(dtype=float)
        m &= np.isfinite(e) & (e >= 0.9)
    return m


# --------------------------------------------------------------------------
# decision: target_transform
#
# The five core anchor targets, in the units each option asks for. sSFR is
# formed as SFR / Mstar before the log, which is AION-1's own convention and
# also removes the mass dependence, so SFR skill can no longer be earned by
# a probe that has only learned mass.
# --------------------------------------------------------------------------

PHYSICAL_TARGETS = ["logMstar", "avgSfr", "tageMw", "zMw", "redshift"]
_LOG_FLOOR = 1e-12


def _safe_log10(x):
    x = np.asarray(x, dtype=float)
    return np.log10(np.where(x > _LOG_FLOOR, x, _LOG_FLOOR))


def _rank_gauss(x):
    from scipy.special import ndtri
    x = np.asarray(x, dtype=float)
    r = pd.Series(x).rank(method="average").to_numpy()
    return ndtri((r - 0.5) / len(r))


def transform_targets(cat: pd.DataFrame, transform: str, targets=PHYSICAL_TARGETS):
    """Return (values dict, label dict) for the requested anchor targets."""
    out, names = {}, {}
    mstar = None
    if "logMstar" in cat.columns:
        mstar = pd.to_numeric(cat["logMstar"], errors="coerce").to_numpy(dtype=float)
    for t in targets:
        if t not in cat.columns:
            continue
        v = pd.to_numeric(cat[t], errors="coerce").to_numpy(dtype=float)
        name = t
        if transform == "identity":
            pass
        elif transform == "rank_gauss":
            v, name = _rank_gauss(v), f"rankGauss({t})"
        elif transform in ("log_skewed", "log_ssfr"):
            if t == "avgSfr":
                if transform == "log_ssfr" and mstar is not None:
                    # log sSFR = log10(SFR) - logMstar
                    v, name = _safe_log10(v) - mstar, "log_sSFR"
                else:
                    v, name = _safe_log10(v), "log_avgSfr"
            elif t == "zMw":
                v, name = _safe_log10(v), "log_zMw"
        else:
            raise ValueError(f"unknown target_transform: {transform}")
        out[t], names[t] = v, name
    return out, names


# ASTRID's exact, particle-derived counterparts of the anchor targets. The
# pairing is what probe transfer is scored against; the mass and SFR column
# choices are themselves decisions (sim_mass_column, sim_sfr_column).
def astrid_targets(cat, transform, mass_col="logMstarAper", sfr_col="sfr100"):
    mass_map = {"log_mstar_aper": "logMstarAper", "log_mstar_total": "logMstarTotal"}
    sfr_map = {"sfr100": "sfr100", "sfr10": "sfr10"}
    mcol = mass_map.get(mass_col, mass_col)
    scol = sfr_map.get(sfr_col, sfr_col)
    num = lambda c: pd.to_numeric(cat[c], errors="coerce").to_numpy(dtype=float)
    mstar = num(mcol)
    out, names = {}, {}
    out["logMstar"], names["logMstar"] = mstar, mcol
    sfr = num(scol)
    if transform == "log_ssfr":
        out["avgSfr"], names["avgSfr"] = _safe_log10(sfr) - mstar, f"log_sSFR({scol})"
    elif transform == "log_skewed":
        out["avgSfr"], names["avgSfr"] = _safe_log10(sfr), f"log({scol})"
    elif transform == "rank_gauss":
        out["avgSfr"], names["avgSfr"] = _rank_gauss(sfr), f"rankGauss({scol})"
    else:
        out["avgSfr"], names["avgSfr"] = sfr, scol
    age = num("massWeightedAgeGyr")
    out["tageMw"], names["tageMw"] = (
        (_rank_gauss(age), "rankGauss(massWeightedAgeGyr)") if transform == "rank_gauss"
        else (age, "massWeightedAgeGyr"))
    met = num("massWeightedMetallicity")
    if transform in ("log_skewed", "log_ssfr"):
        out["zMw"], names["zMw"] = _safe_log10(met), "log(massWeightedMetallicity)"
    elif transform == "rank_gauss":
        out["zMw"], names["zMw"] = _rank_gauss(met), "rankGauss(massWeightedMetallicity)"
    else:
        out["zMw"], names["zMw"] = met, "massWeightedMetallicity"
    z = num("redshift")
    out["redshift"], names["redshift"] = (
        (_rank_gauss(z), "rankGauss(redshift)") if transform == "rank_gauss"
        else (z, "redshift"))
    return out, names


# --------------------------------------------------------------------------
# decision: photometry_baseline_features
#
# The bar the embedding has to clear. gzArm ships fluxes but no magnitudes
# and no sersic index, so magnitudes are formed from flux at the Legacy
# Survey zeropoint of 22.5 and any feature the catalogue lacks is dropped,
# with the surviving feature list recorded on every row of the output.
# --------------------------------------------------------------------------

_BANDS = ["G", "R", "I", "Z"]


def photometry_features(cat: pd.DataFrame, feature_set: str):
    if feature_set not in ("mags_only", "mags_colours", "mags_colours_shape"):
        raise ValueError(f"unknown photometry_baseline_features: {feature_set}")
    cols, names = [], []
    mags = {}
    for b in _BANDS:
        if f"mag{b}" in cat.columns:
            m = pd.to_numeric(cat[f"mag{b}"], errors="coerce").to_numpy(dtype=float)
        elif f"flux{b}" in cat.columns:
            f = pd.to_numeric(cat[f"flux{b}"], errors="coerce").to_numpy(dtype=float)
            m = 22.5 - 2.5 * _safe_log10(f)   # Legacy Survey nanomaggie zeropoint
        else:
            continue
        mags[b] = m
        cols.append(m)
        names.append(f"mag{b}")
    if feature_set in ("mags_colours", "mags_colours_shape"):
        bs = list(mags)
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                cols.append(mags[bs[i]] - mags[bs[j]])
                names.append(f"{bs[i]}-{bs[j]}")
    if feature_set == "mags_colours_shape":
        for c in ("shapeR", "ellip", "ba", "sersic"):
            if c in cat.columns:
                v = pd.to_numeric(cat[c], errors="coerce").to_numpy(dtype=float)
                if c == "shapeR":
                    v = _safe_log10(v)          # spans a decade and a half
                    names.append("log_shapeR")
                else:
                    names.append(c)
                cols.append(v)
    X = np.column_stack(cols)
    return X, names


# --------------------------------------------------------------------------
# decision: nuisance_feature_set
# --------------------------------------------------------------------------

def nuisance_features(cat: pd.DataFrame, feature_set: str):
    groups = {
        "psf_only": ["psfFwhm"],
        "psf_depth": ["psfFwhm", "psfDepth", "galdepth"],
        "psf_depth_ebv": ["psfFwhm", "psfDepth", "galdepth", "ebv", "nobs"],
    }
    if feature_set not in groups:
        raise ValueError(f"unknown nuisance_feature_set: {feature_set}")
    cols, names = [], []
    for pref in groups[feature_set]:
        if pref == "ebv":
            if "ebv" in cat.columns:
                cols.append(pd.to_numeric(cat["ebv"], errors="coerce").to_numpy(dtype=float))
                names.append("ebv")
            continue
        for b in _BANDS:
            c = f"{pref}{b}"
            if c not in cat.columns:
                continue
            v = pd.to_numeric(cat[c], errors="coerce").to_numpy(dtype=float)
            if pref in ("psfDepth", "galdepth", "nobs"):
                v = _safe_log10(v)   # both span three decades across the sample
                names.append(f"log_{c}")
            else:
                names.append(c)
            cols.append(v)
    return np.column_stack(cols), names


# --------------------------------------------------------------------------
# decision: morphology_label_form
# --------------------------------------------------------------------------

MORPH_LABELS = ["spiralArmsFraction", "barStrongFraction", "barWeakFraction",
                "edgeOnFraction", "bulgeDominantFraction", "mergerFraction"]


def morphology_targets(cat: pd.DataFrame, label_form: str, labels=MORPH_LABELS):
    """Yield (label, y, is_binary, threshold) per morphology column.

    Galaxy Zoo gives vote fractions, not labels. A 0.5 threshold is
    degenerate on this sample -- bulgeDominantFraction has a median of 0.011
    and spiralArmsFraction 0.88 -- so threshold_median puts every label on a
    balanced footing at the cost of the threshold no longer meaning "most
    voters agreed".
    """
    for lab in labels:
        if lab not in cat.columns:
            continue
        v = pd.to_numeric(cat[lab], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v)
        if label_form == "vote_fraction_regression":
            yield lab, v, ok, False, np.nan
        elif label_form in ("threshold_half", "threshold_median"):
            thr = 0.5 if label_form == "threshold_half" else float(np.median(v[ok]))
            yield lab, (v > thr).astype(int), ok, True, thr
        else:
            raise ValueError(f"unknown morphology_label_form: {label_form}")


# --------------------------------------------------------------------------
# held-out evaluation
# --------------------------------------------------------------------------

def evaluate_probe(X, y, folds, prep_mode, make_model, task="regression"):
    """Fit and score one probe across the folds of the chosen eval protocol.

    Conditioning is refitted inside every fold on the training rows only, so
    neither the standardisation statistics nor the regularisation strength
    ever see the rows the number is reported on.
    """
    from sklearn.metrics import r2_score, roc_auc_score

    scores, alphas = [], []
    for tr, te in folds:
        prep = EmbeddingPreprocessor(prep_mode).fit(X[tr])
        Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
        model = make_model()
        if task == "classification" and len(np.unique(y[tr])) < 2:
            continue
        model.fit(Xtr, y[tr])
        if task == "classification":
            if len(np.unique(y[te])) < 2:
                continue
            if hasattr(model, "predict_proba"):
                s = model.predict_proba(Xte)[:, 1]
            elif hasattr(model, "decision_function"):
                s = model.decision_function(Xte)
            else:
                s = model.predict(Xte)
            scores.append(roc_auc_score(y[te], s))
        else:
            scores.append(r2_score(y[te], model.predict(Xte)))
        a = getattr(model, "alpha_", None)
        if a is None:
            C = getattr(model, "C_", None)
            a = float(1.0 / np.mean(C)) if C is not None else None
        if a is not None:
            alphas.append(float(np.mean(a)))
    scores = np.asarray(scores, dtype=float)
    return {
        "score_mean": float(scores.mean()) if scores.size else float("nan"),
        "score_std": float(scores.std(ddof=0)) if scores.size else float("nan"),
        "n_folds": int(scores.size),
        "alpha_mean": float(np.mean(alphas)) if alphas else float("nan"),
    }


def fit_full(X, y, prep_mode, make_model, task="regression"):
    """Refit on every row, for the weight vector rather than for a score."""
    prep = EmbeddingPreprocessor(prep_mode).fit(X)
    model = make_model()
    model.fit(prep.transform(X), y)
    return model, prep


# --------------------------------------------------------------------------
# decision: id_estimator x id_scale_selection
# --------------------------------------------------------------------------

def _two_nn(X, discard_frac=0.1):
    """Facco et al.'s TwoNN, as adopted by Ansuini et al. 2019.

    Uses only the ratio mu = r2/r1 of each point's second- to first-nearest
    neighbour distance, which is what lets it tolerate curvature of the
    embedding manifold and local density variation. The top ``discard_frac``
    of the empirical CDF is dropped before the straight-line fit, following
    the original prescription.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 10:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    good = (r1 > 0) & np.isfinite(r2)
    mu = r2[good] / r1[good]
    mu = np.sort(mu[mu > 1.0])
    if mu.size < 10:
        return float("nan")
    # The empirical CDF is defined over the *whole* sorted set and only then
    # truncated. Recomputing F after the cut would put F=1 back at the last
    # retained point, and -log(1-F) is infinite there -- which silently turns
    # every estimate into inf.
    F = np.arange(1, mu.size + 1) / mu.size
    k = int(np.floor(mu.size * (1.0 - discard_frac)))
    if k < 2:
        return float("nan")
    mu, F = mu[:k], F[:k]
    x = np.log(mu)
    yv = -np.log1p(-F)
    # least squares through the origin: the slope is the dimension
    return float(np.dot(x, yv) / np.dot(x, x))


def twonn_sampling_diagnostic(X):
    """Is TwoNN's sampling assumption actually met by this point set?

    TwoNN assumes the points are locally Poisson-distributed. Under that
    assumption mu = r2/r1 has median 2**(1/d), so the median alone implies a
    dimension. A *regularly* sampled set breaks the assumption in a specific,
    recognisable way: every point's two nearest neighbours are nearly
    equidistant, mu collapses towards 1, and the fitted dimension inflates
    without bound. A uniform grid of rotation angles is exactly that case, so
    this diagnostic travels with every per-orbit estimate.
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 3:
        return {}
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    if not np.any(ok):
        return {}
    mu = r2[ok] / r1[ok]
    med = float(np.median(mu))
    implied = float(np.log(2.0) / np.log(med)) if med > 1.0 else float("inf")
    return {"mu_median": med, "dim_implied_by_median_mu": implied,
            "frac_mu_below_1p05": float(np.mean(mu < 1.05))}


def _mle_levina_bickel(X, k=10):
    """Levina-Bickel maximum-likelihood estimator, averaged over points.

    A second nearest-neighbour estimator with a different bias from TwoNN,
    so agreement between the two is meaningful evidence rather than a
    restatement. k is fixed at 10; it is not exposed as a decision.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < k + 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    T = d[:, 1:]
    good = T[:, 0] > 0
    T = T[good]
    if not len(T):
        return float("nan")
    logs = np.log(T[:, -1][:, None] / T[:, :-1])
    m = (logs.mean(axis=1)) ** -1
    m = m[np.isfinite(m)]
    return float(m.mean()) if m.size else float("nan")


def _pca_participation_ratio(X):
    """Effective number of principal components from the eigenvalue spectrum.

    Reported as a contrast, not as a competing estimate: on a curved
    manifold this measures the ambient linear subspace, and Ansuini et al.
    find it one to two orders of magnitude above the TwoNN value.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    if not np.any(lam > 0):
        return float("nan")
    return float(lam.sum() ** 2 / np.square(lam).sum())


def estimate_id(X, estimator):
    if estimator == "twonn":
        return _two_nn(X)
    if estimator == "mle_levina_bickel":
        return _mle_levina_bickel(X)
    if estimator == "pca_participation_ratio":
        return _pca_participation_ratio(X)
    raise ValueError(f"unknown id_estimator: {estimator}")


def _discard_far_pairs(X, frac=0.1):
    """Drop the points with the largest second-neighbour distance.

    Reduces sensitivity to the manifold's global curvature at the cost of an
    added tuning parameter, which is why it is not the default.
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 10:
        return X
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    keep = np.argsort(d[:, 2])[: int(np.floor(len(X) * (1.0 - frac)))]
    return X[keep]


def scale_selected_estimate(estimate_fn, n, scale_selection, seed=SEED, min_block=40):
    """Apply the stability rule to any estimator expressed over a row index.

    ``estimate_fn(idx)`` returns an ID for the rows named by ``idx``. Point
    clouds pass an estimator over their own points; the per-galaxy orbit
    estimate passes one that averages over a subset of galaxies. Either way
    ``block_analysis`` asks the same question: does the number move when the
    sample shrinks?

    A well-defined ID depends only weakly on N, so the reported value is the
    mean over the run of block sizes whose estimates stay within 10% of the
    full-sample value -- the plateau -- and the whole curve is returned so a
    reader can see whether that plateau is real rather than asserted.
    """
    idx_all = np.arange(n)
    if scale_selection == "full_sample":
        return estimate_fn(idx_all), {"mode": "full_sample", "n_points": int(n)}
    if scale_selection == "discard_far_pairs":
        raise ValueError("discard_far_pairs applies to point clouds only")
    if scale_selection != "block_analysis":
        raise ValueError(f"unknown id_scale_selection: {scale_selection}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    curve = []
    s = 1
    while s <= 32 and n // s >= min_block:
        vals = [estimate_fn(order[i::s]) for i in range(s)]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            curve.append({"n_blocks": int(s), "block_size": int(n // s),
                          "id_mean": float(np.mean(vals)),
                          "id_std": float(np.std(vals, ddof=0))})
        s *= 2
    if not curve:
        v = estimate_fn(idx_all)
        return v, {"mode": "block_analysis", "curve": [],
                   "note": f"sample of {n} too small to block below {min_block}; "
                           "reported as a single full-sample estimate"}
    ref = curve[0]["id_mean"]
    stable = []
    for pt in curve:
        if ref > 0 and abs(pt["id_mean"] - ref) / ref <= 0.10:
            stable.append(pt["id_mean"])
        else:
            break
    reported = float(np.mean(stable)) if stable else float(ref)
    return reported, {"mode": "block_analysis", "curve": curve,
                      "n_stable_blocks": len(stable),
                      "full_sample_id": float(ref),
                      "plateau_tolerance": 0.10}


def id_with_scale_selection(X, estimator, scale_selection, seed=SEED):
    """Stability-checked ID of a point cloud."""
    X = np.asarray(X, dtype=np.float64)
    if scale_selection == "discard_far_pairs":
        Xd = _discard_far_pairs(X)
        return estimate_id(Xd, estimator), {
            "mode": "discard_far_pairs", "discard_frac": 0.1,
            "n_points": int(len(Xd))}
    return scale_selected_estimate(
        lambda idx: estimate_id(X[idx], estimator), len(X), scale_selection, seed)


ID_UNDERESTIMATION_THRESHOLD = 20.0   # Ansuini et al.: above this it is a bound


def id_report(value, diag, sampling_is_regular=False):
    """Attach the caveat that actually applies to this number.

    Two different caveats are in play and they point opposite ways. On a
    fairly sampled cloud, TwoNN *under*estimates above ID ~20, so a large
    value is a lower bound. On a regularly sampled set it *over*estimates
    without bound, so the value is not an estimate at all. Reporting the
    first caveat where the second applies would be worse than reporting none.
    """
    lower_bound = bool(np.isfinite(value) and value > ID_UNDERESTIMATION_THRESHOLD)
    if sampling_is_regular:
        caveat = (
            "Not a dimension estimate. These points are sampled on a uniform "
            "grid, which violates TwoNN's local-Poisson assumption: each "
            "point's two nearest neighbours are near-equidistant, mu collapses "
            "towards 1 and the fitted value inflates without bound. Read the "
            "harmonic spectrum and the linear participation ratio instead."
        )
        lower_bound = False
    elif lower_bound:
        caveat = (
            "Above ID ~20 TwoNN moderately underestimates at finite sample "
            "size, especially where density is non-uniform, so this is a lower "
            "bound, not an estimate."
        )
    else:
        caveat = (
            "Below ID ~20 the estimate stays close to ground truth at finite "
            "sample size, so this is reported as an estimate."
        )
    return {
        "id": float(value),
        "is_lower_bound": lower_bound,
        "sampling_assumption_violated": bool(sampling_is_regular),
        "caveat": caveat,
        "diagnostic": diag,
    }


# --------------------------------------------------------------------------
# Fast exact ridge with an inner-CV alpha grid.
#
# sklearn's RidgeCV(cv=5) refits from scratch for every (alpha, fold) pair,
# which at 1024 features and a 28-point grid dominates the whole analysis.
# The penalty only enters through the eigenvalues of X'X, so one
# eigendecomposition per fold serves the entire grid. The selected alpha and
# the fitted coefficients are identical to RidgeCV(cv=5); only the cost is
# different.
# --------------------------------------------------------------------------

class SvdRidgeCV:
    """Ridge whose penalty is chosen by k-fold CV inside the training rows."""

    def __init__(self, alphas=None, cv=5, seed=SEED):
        self.alphas = np.asarray(ALPHA_GRID if alphas is None else alphas, dtype=float)
        self.cv = cv
        self.seed = seed

    @staticmethod
    def _prepare(X, y):
        xm, ym = X.mean(axis=0), float(np.mean(y))
        Xc, yc = X - xm, y - ym
        A = Xc.T @ Xc
        b = Xc.T @ yc
        # A is symmetric PSD, so eigh is both faster and better conditioned
        # here than a general SVD of Xc.
        evals, evecs = np.linalg.eigh(A)
        evals = np.clip(evals, 0.0, None)
        return xm, ym, evecs, evals, evecs.T @ b

    @staticmethod
    def _coef(evecs, evals, vtb, alpha):
        return evecs @ (vtb / (evals + alpha))

    def fit(self, X, y):
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        scores = np.zeros(len(self.alphas))
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            xm, ym, evecs, evals, vtb = self._prepare(X[tr], y[tr])
            Xva = X[va] - xm
            for i, alpha in enumerate(self.alphas):
                pred = Xva @ self._coef(evecs, evals, vtb, alpha) + ym
                scores[i] += r2_score(y[va], pred)
        self.cv_scores_ = scores / self.cv
        self.alpha_ = float(self.alphas[int(np.argmax(self.cv_scores_))])

        xm, ym, evecs, evals, vtb = self._prepare(X, y)
        self.coef_ = self._coef(evecs, evals, vtb, self.alpha_)
        self.intercept_ = ym - float(xm @ self.coef_)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------
# decision: quotient_operator
#
# Defines what "removing rotation" means, and therefore what the headline
# residual dimensionality is a dimensionality *of*. The options differ in how
# aggressive they are, so the operator travels with every number it produces.
# --------------------------------------------------------------------------

def quotient_representation(orbits, operator, full_cloud=None):
    """Map measured rotation orbits to a rotation-invariant representation.

    ``orbits`` is (n_galaxies, n_angles, d). Returns (Q, info) where Q is the
    quotiented cloud and info records how it was formed.
    """
    orbits = np.asarray(orbits, dtype=np.float64)
    g, a, d = orbits.shape
    if operator == "orbit_mean":
        # Exactly the zeroth angular harmonic: the simplest rotation-invariant
        # summary, keeping everything rotation cannot change.
        return orbits.mean(axis=1), {"operator": operator, "dim": d,
                                     "applies_to_full_cloud": False}
    if operator == "harmonic_magnitudes":
        # Keep harmonic magnitudes, discard their phases: still rotation
        # invariant, but retains angular structure the plain mean throws away,
        # so the residual dimension should come out higher.
        F = np.fft.rfft(orbits, axis=1)
        Q = np.abs(F).reshape(g, -1)
        return Q, {"operator": operator, "dim": Q.shape[1],
                   "n_harmonics": F.shape[1], "applies_to_full_cloud": False}
    if operator == "pca_deflation":
        # Pool within-orbit variation across galaxies, take its leading
        # directions and project them out. The most aggressive option: it also
        # removes directions that merely correlate with orientation.
        R = (orbits - orbits.mean(axis=1, keepdims=True)).reshape(g * a, d)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        var = S ** 2
        frac = np.cumsum(var) / var.sum()
        k = int(np.searchsorted(frac, 0.90) + 1)
        P = Vt[:k]
        X = orbits.mean(axis=1) if full_cloud is None else np.asarray(full_cloud, float)
        Q = X - (X @ P.T) @ P
        return Q, {"operator": operator, "dim": d, "n_deflated": k,
                   "deflated_variance_fraction": 0.90,
                   "applies_to_full_cloud": full_cloud is not None}
    raise ValueError(f"unknown quotient_operator: {operator}")


# --------------------------------------------------------------------------
# decisions: domain_matching x matching_method
#
# The real and simulated samples were drawn by different procedures. Without
# matching, a classifier reaches near-perfect separation by noticing that
# simulated galaxies sit at different masses -- which says nothing about
# whether the model represents them differently.
# --------------------------------------------------------------------------

def matching_variables(real_cat, sim_cat, domain_matching, sim_mass_column):
    """Build the aligned matching variables for the two samples."""
    if domain_matching == "none":
        return None, None, []
    mass_map = {"log_mstar_aper": "logMstarAper", "log_mstar_total": "logMstarTotal"}
    smass = mass_map.get(sim_mass_column, sim_mass_column)
    num = lambda c, d: pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)

    cols = [("logMstar", smass, "log_mstar")]
    if domain_matching in ("mass_redshift", "mass_redshift_size"):
        cols.append(("redshift", "redshift", "redshift"))
    if domain_matching == "mass_redshift_size":
        # ASTRID records no apparent size. enclR is the fraction of light
        # inside the frame, which is a framing statistic, not a size, so
        # there is nothing honest to match against here.
        raise SystemExit(
            "domain_matching 'mass_redshift_size' needs an apparent-size column "
            "for the simulated sample; the ASTRID catalogue ships none "
            "(enclR is enclosed-light fraction, not size). Use mass_redshift."
        )
    if domain_matching not in ("mass_only", "mass_redshift"):
        raise ValueError(f"unknown domain_matching: {domain_matching}")

    R = np.column_stack([num(rc, real_cat) for rc, _, _ in cols])
    S = np.column_stack([num(sc, sim_cat) for _, sc, _ in cols])
    return R, S, [n for _, _, n in cols]


def match_samples(R, S, method, seed=SEED):
    """Return (real_idx, sim_idx, weights, info) equalising the two samples.

    Matching happens in the z-scored space of the matching variables, pooled
    across both samples so neither defines the scale on its own.
    """
    pool = np.vstack([R, S])
    mu, sd = pool.mean(axis=0), pool.std(axis=0)
    sd[sd == 0] = 1.0
    Rz, Sz = (R - mu) / sd, (S - mu) / sd

    if method == "nearest_neighbour_pairs":
        # Pair each simulated galaxy with the closest unused real galaxy,
        # discarding unmatched rows. Exactly balanced, so the resulting AUC
        # needs no weighting to interpret.
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(Sz))
        tree = cKDTree(Rz)
        used = np.zeros(len(Rz), dtype=bool)
        r_idx, s_idx, dists = [], [], []
        k = min(len(Rz), 50)
        dd, ii = tree.query(Sz[order], k=k)
        # cKDTree drops the neighbour axis when k == 1; reshape rather than
        # atleast_2d, which would transpose it.
        dd, ii = np.reshape(dd, (-1, k)), np.reshape(ii, (-1, k))
        for row, si in enumerate(order):
            for dist, cand in zip(dd[row], ii[row]):
                if not used[cand]:
                    used[cand] = True
                    r_idx.append(cand); s_idx.append(si); dists.append(dist)
                    break
        r_idx, s_idx = np.array(r_idx, int), np.array(s_idx, int)
        w = np.ones(len(r_idx))
        return r_idx, s_idx, (w, w), {
            "method": method, "n_pairs": int(len(r_idx)),
            "mean_pair_distance_sd_units": float(np.mean(dists)) if dists else float("nan"),
            "max_neighbours_searched": int(k),
        }

    if method == "histogram_reweight":
        # Bin the matching variables and reweight so the marginals agree.
        # Retains all rows; the effective sample size can be far below the
        # nominal count, so it is reported alongside.
        bins = [np.quantile(np.vstack([Rz, Sz])[:, j], np.linspace(0, 1, 11))
                for j in range(Rz.shape[1])]
        def cell(Z):
            return tuple(np.digitize(Z[:, j], bins[j][1:-1]) for j in range(Z.shape[1]))
        rc, sc = cell(Rz), cell(Sz)
        rkey = np.ravel_multi_index(rc, [11] * Rz.shape[1])
        skey = np.ravel_multi_index(sc, [11] * Sz.shape[1])
        rcount = np.bincount(rkey, minlength=11 ** Rz.shape[1]).astype(float)
        scount = np.bincount(skey, minlength=11 ** Sz.shape[1]).astype(float)
        target = np.minimum(rcount, scount)
        wr = np.divide(target, rcount, out=np.zeros_like(rcount), where=rcount > 0)[rkey]
        ws = np.divide(target, scount, out=np.zeros_like(scount), where=scount > 0)[skey]
        ess = lambda w: float(w.sum() ** 2 / np.square(w).sum()) if np.any(w) else 0.0
        return (np.arange(len(Rz)), np.arange(len(Sz)), (wr, ws),
                {"method": method, "effective_n_real": ess(wr),
                 "effective_n_sim": ess(ws)})

    if method == "propensity_score":
        # Match on the fitted probability of being simulated. Handles several
        # variables at once, but only equalises the score, not the variables.
        from scipy.spatial import cKDTree
        from sklearn.linear_model import LogisticRegression
        X = np.vstack([Rz, Sz])
        y = np.r_[np.zeros(len(Rz)), np.ones(len(Sz))]
        ps = LogisticRegression(max_iter=2000).fit(X, y).predict_proba(X)[:, 1]
        pr, psim = ps[:len(Rz)][:, None], ps[len(Rz):][:, None]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(psim))
        tree = cKDTree(pr)
        used = np.zeros(len(pr), dtype=bool)
        r_idx, s_idx = [], []
        k = min(len(pr), 50)
        _, ii = tree.query(psim[order], k=k)
        ii = np.reshape(ii, (-1, k))
        for row, si in enumerate(order):
            for cand in ii[row]:
                if not used[cand]:
                    used[cand] = True
                    r_idx.append(cand); s_idx.append(si)
                    break
        r_idx, s_idx = np.array(r_idx, int), np.array(s_idx, int)
        w = np.ones(len(r_idx))
        return r_idx, s_idx, (w, w), {"method": method, "n_pairs": int(len(r_idx))}

    raise ValueError(f"unknown matching_method: {method}")


# --------------------------------------------------------------------------
# decisions: nuisance_projection x nuisance_subspace_dim
# --------------------------------------------------------------------------

def nuisance_basis(nuisance_csv, dim, projection, prep=None):
    """Orthonormal basis of the instrument subspace, in the conditioned space.

    The probe directions are stored as functionals on the raw embedding, so
    they are re-expressed for whatever conditioning the caller is using before
    the SVD. The requested dimension is capped at the rank actually available:
    asking for more directions than the probes span cannot manufacture them,
    and the artifact records the cap rather than hiding it.
    """
    if projection == "none":
        return None, {"nuisance_projection": "none", "n_directions": 0}
    if projection == "cca_subspace":
        raise SystemExit(
            "nuisance_projection 'cca_subspace' needs the nuisance variables "
            "themselves, not the probe directions; probes.nuisance_probe_r2 "
            "ships directions only. Use probe_weight_span."
        )
    if projection != "probe_weight_span":
        raise ValueError(f"unknown nuisance_projection: {projection}")

    df = pd.read_csv(nuisance_csv)
    wcols = sorted(c for c in df.columns if c.startswith("w") and c[1:].isdigit())
    if not wcols:
        raise SystemExit(
            f"{nuisance_csv} carries no probe weight columns; "
            "domain_gap.residual_discriminability needs the fitted directions, "
            "not only the R2 values."
        )
    W = df[wcols].to_numpy(dtype=np.float64)          # (n_vars, d), raw space
    if prep is not None:
        W = np.vstack([prep.direction_to_transformed(w) for w in W])
    n = np.linalg.norm(W, axis=1, keepdims=True)
    n[n == 0] = 1.0
    W = W / n

    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    want = int(str(dim).lstrip("d")) if isinstance(dim, str) else int(dim)
    tol = S.max() * max(W.shape) * np.finfo(float).eps
    rank = int(np.sum(S > tol))
    k = min(want, rank)
    return Vt[:k], {
        "nuisance_projection": projection,
        "nuisance_subspace_dim_requested": want,
        "n_directions": k,
        "probe_directions_available": int(W.shape[0]),
        "probe_span_rank": rank,
        "capped_at_available_rank": bool(k < want),
        "singular_values": [float(v) for v in S[:k]],
        "nuisance_variables": df["variable"].tolist() if "variable" in df else [],
        "nuisance_probe_r2": (dict(zip(df["variable"], df["r2_mean"]))
                              if {"variable", "r2_mean"} <= set(df.columns) else {}),
    }


def project_out(X, basis):
    """Remove the span of ``basis`` (orthonormal rows) from the rows of X."""
    if basis is None or len(basis) == 0:
        return X
    return X - (X @ basis.T) @ basis


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def by_output_id(paths):
    """Index a list of `{inputs}` paths by the output id each directory names."""
    return {Path(p).name: Path(p) for p in paths}


def load_csv(d):
    return pd.read_csv(resolve_input(d, "csv"))


def load_json(d):
    return json.loads(resolve_input(d, "json").read_text())


def figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8,
    })
    return plt
