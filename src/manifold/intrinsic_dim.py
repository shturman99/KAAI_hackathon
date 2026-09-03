"""Intrinsic dimensionality of the embedding cloud, of one rotation orbit,
and of what survives quotienting rotation out.

Three modes behind one script, because they must share an estimator, a
stability rule and -- crucially -- a metric. Whenever the anchor cloud is
available the conditioning is fitted on it and applied to everything else, so
the full, orbit and quotient numbers are dimensions measured in the *same*
space and the dimensionality budget they form is meaningful.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc


def load_orbits(path):
    d = Path(path)
    orbits = np.load(lc.resolve_input(path, "npy"))
    idx = d / "orbit_row_index.npy"
    return orbits, (np.load(idx) if idx.is_file() else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["full", "orbit", "quotient", "matched"])
    ap.add_argument("--n-draws", type=int, default=64)
    ap.add_argument("--emb")
    ap.add_argument("--orbits")
    ap.add_argument("--cat")
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--real-quality-cuts")
    ap.add_argument("--id-estimator", required=True)
    # Not required in matched mode: that mode does its own subsampling at a
    # fixed n, so there is no scale to select and the decision is deliberately
    # not declared on the output.
    ap.add_argument("--id-scale-selection")
    ap.add_argument("--quotient-operator")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.mode != "matched" and not a.id_scale_selection:
        ap.error(f"--id-scale-selection is required for --mode {a.mode}")

    meta = {
        "mode": a.mode,
        "id_estimator": a.id_estimator,
        "embedding_preprocessing": a.embedding_preprocessing,
    }
    if a.id_scale_selection:
        meta["id_scale_selection"] = a.id_scale_selection

    # ---- the anchor cloud, and the metric everything is measured in --------
    anchor = prep = None
    if a.emb:
        anchor = np.load(lc.resolve_input(a.emb, "npy")).astype(np.float64)
        mask = np.ones(len(anchor), dtype=bool)
        if a.real_quality_cuts and a.cat:
            cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
            mask = lc.real_cut_mask(cat, a.real_quality_cuts)
            meta["real_quality_cuts"] = a.real_quality_cuts
        elif a.real_quality_cuts:
            meta["real_quality_cuts"] = a.real_quality_cuts
            meta["real_quality_cuts_applied"] = False
        anchor = anchor[mask]
        meta["n_anchor_rows"] = int(len(anchor))
        prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing).fit(anchor)

    # ---- mode: the full cloud as shipped ----------------------------------
    if a.mode == "full":
        if anchor is None:
            raise SystemExit("--mode full needs --emb")
        X = prep.transform(anchor)
        value, diag = lc.id_with_scale_selection(X, a.id_estimator, a.id_scale_selection)
        payload = lc.id_report(value, diag)
        payload.update(meta)
        payload["n_points"] = int(len(X))
        payload["ambient_dim"] = int(X.shape[1])

    # ---- mode: one galaxy's rotation orbit --------------------------------
    elif a.mode == "orbit":
        if not a.orbits:
            raise SystemExit("--mode orbit needs --orbits")
        orbits, _ = load_orbits(a.orbits)
        g, n_ang, d = orbits.shape
        if prep is None:
            # No anchor cloud declared for this output: condition on the
            # pooled orbit points instead, and say so in the artifact.
            prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing).fit(
                orbits.reshape(g * n_ang, d))
            meta["conditioned_on"] = "pooled orbit points"
        else:
            meta["conditioned_on"] = "anchor cloud"
        Z = prep.transform(orbits.reshape(g * n_ang, d)).reshape(g, n_ang, d)

        # One ID per galaxy over its own n_angles points; the reported value
        # is their mean. The stability rule then runs over *galaxies*, which
        # is the sampling question the small orbit subsample actually raises.
        per_galaxy = np.array([lc.estimate_id(Z[i], a.id_estimator) for i in range(g)])
        finite = np.isfinite(per_galaxy)
        value, diag = lc.scale_selected_estimate(
            lambda idx: float(np.mean(per_galaxy[idx][np.isfinite(per_galaxy[idx])]))
            if np.any(np.isfinite(per_galaxy[idx])) else float("nan"),
            g, a.id_scale_selection, min_block=32,
        )
        # A rotation orbit is sampled on a uniform angle grid by construction,
        # so TwoNN's local-Poisson assumption is violated here on purpose. The
        # diagnostic says how badly, and the linear participation ratio gives a
        # reading that does not rest on that assumption at all: a pure circle
        # spans a 2-plane, so a value near 2 means the orbit really is one
        # closed curve, whatever the neighbour-ratio estimator reports.
        diagnostics = [lc.twonn_sampling_diagnostic(Z[i]) for i in range(g)]
        mu_med = np.array([d_.get("mu_median", np.nan) for d_ in diagnostics])
        pr = np.array([lc._pca_participation_ratio(Z[i]) for i in range(g)])
        regular = bool(np.nanmedian(mu_med) < 1.3 and a.id_estimator == "twonn")

        payload = lc.id_report(value, diag, sampling_is_regular=regular)
        payload.update(meta)
        payload.update({
            "n_galaxies": int(g),
            "n_angles": int(n_ang),
            "ambient_dim": int(d),
            "per_galaxy_id_mean": float(per_galaxy[finite].mean()),
            "per_galaxy_id_std": float(per_galaxy[finite].std(ddof=0)),
            "per_galaxy_id_median": float(np.median(per_galaxy[finite])),
            "n_galaxies_estimated": int(finite.sum()),
            "mu_median_over_galaxies": float(np.nanmedian(mu_med)),
            "mu_median_expected_for_1d": 2.0,
            "linear_participation_ratio_mean": float(np.nanmean(pr)),
            "linear_participation_ratio_std": float(np.nanstd(pr)),
            "participation_ratio_note":
                "a single closed circle spans a 2-plane, so PR near 2 means the "
                "orbit is one closed curve; PR well above 2 means rotation "
                "sweeps out more than a simple circle",
        })

    # ---- mode: full vs quotient at a matched sample size -------------------
    # TwoNN is sample-size dependent, and the full cloud (8058 points) and the
    # quotient (one point per orbit, so orbit_subsample points) are measured at
    # very different n. Comparing their headline values overstates the drop by
    # whatever finite-sample bias sits between the two sizes. Here the full
    # cloud is subsampled down to the quotient's own n and re-estimated many
    # times, so the difference reported is attributable to the quotient rather
    # than to n.
    elif a.mode == "matched":
        if anchor is None or not a.orbits or not a.quotient_operator:
            raise SystemExit("--mode matched needs --emb, --orbits and "
                             "--quotient-operator")
        orbits, _ = load_orbits(a.orbits)
        Q, qinfo = lc.quotient_representation(orbits, a.quotient_operator,
                                              full_cloud=anchor)
        if Q.shape[1] != anchor.shape[1]:
            raise SystemExit(
                f"matched mode needs the quotient to live in the anchor cloud's "
                f"ambient space; quotient_operator={a.quotient_operator} gives "
                f"{Q.shape[1]}d against the cloud's {anchor.shape[1]}d")
        X = prep.transform(anchor)
        Zq = prep.transform(Q)
        n = int(len(Zq))

        rng = np.random.default_rng(lc.SEED)
        draws = np.array([
            lc.estimate_id(X[rng.choice(len(X), size=n, replace=False)],
                           a.id_estimator)
            for _ in range(a.n_draws)
        ])
        draws = draws[np.isfinite(draws)]
        q_id = float(lc.estimate_id(Zq, a.id_estimator))
        full_id = float(lc.estimate_id(X, a.id_estimator))

        payload = dict(meta)
        payload.update({
            "quotient_operator": a.quotient_operator,
            "quotient": qinfo,
            "conditioned_on": "anchor cloud",
            "ambient_dim": int(X.shape[1]),
            "matched_n": n,
            "n_draws": int(draws.size),
            "full_cloud_id_at_matched_n_mean": float(draws.mean()),
            "full_cloud_id_at_matched_n_std": float(draws.std(ddof=0)),
            "full_cloud_id_at_matched_n_p05": float(np.percentile(draws, 5)),
            "full_cloud_id_at_matched_n_p95": float(np.percentile(draws, 95)),
            "quotient_id_at_matched_n": q_id,
            "delta_matched": float(draws.mean() - q_id),
            "full_cloud_id_at_full_n": full_id,
            "n_full_cloud": int(len(X)),
            "delta_naive": float(full_id - q_id),
            "finite_sample_bias": float(full_id - draws.mean()),
            "id": float(draws.mean() - q_id),
            "is_lower_bound": False,
            "interpretation":
                "delta_matched is the dimensionality the quotient removes once "
                "sample size is held fixed. delta_naive compares the two at "
                "their native sample sizes and is inflated by "
                "finite_sample_bias, which is TwoNN's own drift between n_full "
                "and matched_n on this cloud.",
        })
        print(f"  matched: full@n={n} {draws.mean():.3f} +/- {draws.std(ddof=0):.3f}"
              f" vs quotient {q_id:.3f} -> delta {draws.mean() - q_id:.3f}"
              f" (naive delta {full_id - q_id:.3f})", file=sys.stderr)
        lc.write_metric(a.out, payload)
        return 0

    # ---- mode: what survives the quotient ---------------------------------
    else:
        if not a.orbits or not a.quotient_operator:
            raise SystemExit("--mode quotient needs --orbits and --quotient-operator")
        orbits, _ = load_orbits(a.orbits)
        Q, qinfo = lc.quotient_representation(orbits, a.quotient_operator,
                                              full_cloud=anchor)
        meta["quotient_operator"] = a.quotient_operator
        if prep is not None and Q.shape[1] == anchor.shape[1]:
            # Same ambient space as the anchor cloud, so the same conditioning
            # applies and full-vs-quotient is a like-for-like comparison.
            Z = prep.transform(Q)
            meta["conditioned_on"] = "anchor cloud"
        else:
            Z = lc.EmbeddingPreprocessor(a.embedding_preprocessing).fit_transform(Q)
            meta["conditioned_on"] = "quotiented cloud (different ambient dim)"
        value, diag = lc.id_with_scale_selection(Z, a.id_estimator, a.id_scale_selection)
        payload = lc.id_report(value, diag)
        payload.update(meta)
        payload["quotient"] = qinfo
        payload["n_points"] = int(len(Z))
        payload["ambient_dim"] = int(Z.shape[1])

    print(f"  {a.mode}: ID = {payload['id']:.3f}"
          f"{'  (lower bound)' if payload['is_lower_bound'] else ''}", file=sys.stderr)
    lc.write_metric(a.out, payload)


if __name__ == "__main__":
    sys.exit(main())
