"""Real-versus-simulated discriminability, at three stages of control.

ASTRID images carry no PSF and no noise, so a naive classifier here is a
noise detector, not a physics result. The three stages this script produces
are cumulative:

  raw       no controls. Expected near 1, and reported as the reference point
            the later numbers are measured against, not as a finding.
  matched   after the samples are matched in mass and redshift, so the
            classifier cannot win on a population difference.
  residual  additionally after the instrument subspace recovered by
            probes.nuisance_probe_r2 is projected out of both embeddings.

The residual number is the headline for goal 3, and it carries a hard limit:
projection removes the component of the instrument signature that the
nuisance probes capture *linearly*. It does not remove signature that is
nonlinearly encoded, and it is not a substitute for degrading the simulated
images with a matched PSF and injected noise -- which was considered for this
project and declined on time grounds. That is the central limitation of goal
3 and it is written into every artifact this script produces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc

LIMITATION = (
    "Simulated images were not degraded to match the real PSF and noise; the "
    "confound is controlled by distribution matching plus projection of the "
    "linearly-captured instrument subspace only. Residual separation may still "
    "contain instrument signature that is nonlinearly encoded."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--real-cat")
    ap.add_argument("--sim-cat")
    ap.add_argument("--nuisance")
    ap.add_argument("--stage", required=True, choices=["raw", "matched", "residual"])
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--real-quality-cuts", required=True)
    ap.add_argument("--sim-quality-cuts", required=True)
    ap.add_argument("--eval-split", required=True)
    ap.add_argument("--classifier-model", required=True)
    ap.add_argument("--domain-matching", default="none")
    ap.add_argument("--matching-method", default="nearest_neighbour_pairs")
    ap.add_argument("--sim-mass-column", default="log_mstar_aper")
    ap.add_argument("--nuisance-projection", default="none")
    ap.add_argument("--nuisance-subspace-dim", default="d10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Xr = np.load(lc.resolve_input(a.real, "npy")).astype(np.float64)
    Xs = np.load(lc.resolve_input(a.sim, "npy")).astype(np.float64)

    info = {
        "stage": a.stage,
        "classifier_model": a.classifier_model,
        "eval_split": a.eval_split,
        "embedding_preprocessing": a.embedding_preprocessing,
        "real_quality_cuts": a.real_quality_cuts,
        "sim_quality_cuts": a.sim_quality_cuts,
        "limitation": LIMITATION,
        "classifier_penalty": "fixed (C=1.0); probe_regularization is not a "
                              "declared decision for this output",
    }

    # ---- quality cuts -----------------------------------------------------
    real_cat = pd.read_parquet(lc.resolve_input(a.real_cat, "parquet")) if a.real_cat else None
    sim_cat = pd.read_parquet(lc.resolve_input(a.sim_cat, "parquet")) if a.sim_cat else None
    rmask = (lc.real_cut_mask(real_cat, a.real_quality_cuts) if real_cat is not None
             else np.ones(len(Xr), bool))
    smask = (lc.sim_cut_mask(sim_cat, a.sim_quality_cuts) if sim_cat is not None
             else np.ones(len(Xs), bool))
    if real_cat is None or sim_cat is None:
        info["quality_cuts_applied"] = False
    r_rows, s_rows = np.flatnonzero(rmask), np.flatnonzero(smask)
    info["n_real_after_cuts"] = int(r_rows.size)
    info["n_sim_after_cuts"] = int(s_rows.size)

    # ---- matching ---------------------------------------------------------
    wr = ws = None
    if a.stage == "raw":
        info["domain_matching"] = "none"
    else:
        if real_cat is None or sim_cat is None:
            raise SystemExit(f"--stage {a.stage} needs --real-cat and --sim-cat")
        R, S, mvars = lc.matching_variables(
            real_cat.iloc[r_rows], sim_cat.iloc[s_rows],
            a.domain_matching, a.sim_mass_column)
        if R is not None:
            fin = np.isfinite(R).all(axis=1)
            fins = np.isfinite(S).all(axis=1)
            r_rows, s_rows = r_rows[fin], s_rows[fins]
            ri, si, (wr, ws), minfo = lc.match_samples(R[fin], S[fins], a.matching_method)
            r_rows, s_rows = r_rows[ri], s_rows[si]
            info["domain_matching"] = a.domain_matching
            info["matching_variables"] = mvars
            info["matching"] = minfo
            info["sim_mass_column"] = a.sim_mass_column
        else:
            info["domain_matching"] = "none"

    Xr_u, Xs_u = Xr[r_rows], Xs[s_rows]
    info["n_real_used"] = int(len(Xr_u))
    info["n_sim_used"] = int(len(Xs_u))

    X = np.vstack([Xr_u, Xs_u])
    y = np.r_[np.zeros(len(Xr_u), int), np.ones(len(Xs_u), int)]
    w = np.r_[wr, ws] if wr is not None else None

    # ---- nuisance projection ---------------------------------------------
    # Conditioning is fitted on the pooled sample first, so the probe
    # directions are re-expressed in the same space the classifier sees.
    prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing).fit(X)
    basis = None
    if a.stage == "residual":
        if not a.nuisance:
            raise SystemExit("--stage residual needs --nuisance")
        ncsv = lc.resolve_input(a.nuisance, "csv")
        basis, ninfo = lc.nuisance_basis(ncsv, a.nuisance_subspace_dim,
                                         a.nuisance_projection, prep=prep)
        info.update(ninfo)
    else:
        info["nuisance_projection"] = "none"

    # ---- held-out AUC -----------------------------------------------------
    from sklearn.metrics import roc_auc_score
    folds = lc.make_folds(len(X), a.eval_split, stratify=y)
    aucs = []
    for tr, te in folds:
        p = lc.EmbeddingPreprocessor(a.embedding_preprocessing).fit(X[tr])
        Ztr, Zte = p.transform(X[tr]), p.transform(X[te])
        if basis is not None:
            b, _ = lc.nuisance_basis(ncsv, a.nuisance_subspace_dim,
                                     a.nuisance_projection, prep=p)
            Ztr, Zte = lc.project_out(Ztr, b), lc.project_out(Zte, b)
        # These outputs declare classifier_model but not probe_regularization,
        # so the penalty is not a decision here and is held fixed rather than
        # tuned invisibly. It is recorded in the artifact. AUC in this regime
        # is weakly sensitive to it, and an inner grid search at 1024 features
        # over three stages and five folds would dominate the whole analysis.
        clf = lc.make_classifier(a.classifier_model, "fixed")
        if w is not None:
            try:
                clf.fit(Ztr, y[tr], sample_weight=w[tr])
            except TypeError:
                clf.fit(Ztr, y[tr])
        else:
            clf.fit(Ztr, y[tr])
        s = (clf.predict_proba(Zte)[:, 1] if hasattr(clf, "predict_proba")
             else clf.decision_function(Zte))
        aucs.append(roc_auc_score(y[te], s,
                                  sample_weight=w[te] if w is not None else None))

    aucs = np.asarray(aucs, float)
    info["auc_mean"] = float(aucs.mean())
    info["auc_std"] = float(aucs.std(ddof=0))
    info["auc_folds"] = [float(v) for v in aucs]
    info["n_folds"] = int(aucs.size)

    print(f"  {a.stage}: AUC = {info['auc_mean']:.4f} +/- {info['auc_std']:.4f} "
          f"(n_real={info['n_real_used']}, n_sim={info['n_sim_used']})", file=sys.stderr)
    lc.write_metric(a.out, info)


if __name__ == "__main__":
    sys.exit(main())
