"""Separate genuine angular structure from resampling artefact.

The orbit spectrum puts 17.9% of the varying power at harmonic 4 -- the
90-degree mode. That is exactly the symmetry of the square pixel grid and of
the square crop, so a 4-fold signal is precisely what an interpolation
artefact would look like, and the spectrum on its own cannot tell the two
apart.

This control can: ``encode_orbits`` routes exact multiples of 90 degrees
through ``np.rot90``, a pure index permutation that resamples nothing, while
every other angle goes through the chosen interpolation kernel. On the 16-angle
grid that splits the orbit into four sub-orbits of four frames each, all
spanning 360 degrees at identical 90-degree spacing:

    phase 0 -> 0, 90, 180, 270      exact, no interpolation
    phase 1 -> 22.5, 112.5, ...     interpolated
    phase 2 -> 45, 135, ...         interpolated
    phase 3 -> 67.5, 157.5, ...     interpolated

Geometrically the four are the same measurement. The only difference is
whether the pixels were resampled. So the variation carried by phase 0 is
encoder response to orientation that no interpolation could have manufactured,
and the excess variation in phases 1-3 is the resampling cost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc


def _varying_fraction(S):
    """Fraction of a sub-orbit's power that varies across its frames.

    ``S`` is (n_galaxies, n_frames, d). The mean over frames is the
    rotation-invariant part; everything else is what rotation moves. Returned
    per galaxy so the spread over the sample is available downstream.
    """
    mean = S.mean(axis=1, keepdims=True)
    varying = ((S - mean) ** 2).sum(axis=(1, 2))
    total = (S ** 2).sum(axis=(1, 2))
    return np.divide(varying, total, out=np.zeros_like(varying), where=total > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orbits", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--rotation-interpolation", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d_in = Path(lc.resolve_input(a.orbits, "npy")).parent
    orbits = np.load(lc.resolve_input(a.orbits, "npy")).astype(np.float64)
    angles = np.load(d_in / "orbit_angles_deg.npy")
    g, n_ang, d = orbits.shape

    # Four frames 90 degrees apart is the coarsest grid that still closes the
    # orbit, so the control needs the angle grid to be a multiple of 4.
    if n_ang % 4:
        raise SystemExit(f"exact-rotation control needs n_angles divisible by 4, "
                         f"got {n_ang}")
    step = n_ang // 4
    exact_idx = np.arange(4) * step
    if not np.allclose(angles[exact_idx] % 90, 0):
        raise SystemExit(f"phase-0 frames are not exact 90-degree multiples: "
                         f"{angles[exact_idx]}")

    # Conditioning is fitted on the pooled orbit points, matching
    # orbit_spectrum, so the two outputs describe the same space.
    prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing)
    Z = prep.fit_transform(orbits.reshape(g * n_ang, d)).reshape(g, n_ang, d)

    rows = []
    per_phase = {}
    for phase in range(step):
        idx = phase + np.arange(4) * step
        frac = _varying_fraction(Z[:, idx, :])
        per_phase[phase] = frac
        rows.append({
            "phase": phase,
            "angles_deg": "|".join(f"{x:g}" for x in angles[idx]),
            "resampled": bool(phase != 0),
            "varying_power_frac_mean": float(frac.mean()),
            "varying_power_frac_std": float(frac.std(ddof=0)),
            "varying_power_frac_median": float(np.median(frac)),
            "n_galaxies": int(g),
            "n_frames": 4,
        })

    df = pd.DataFrame(rows)
    exact = per_phase[0]
    interp = np.concatenate([per_phase[p] for p in range(1, step)]) if step > 1 \
        else np.array([np.nan])

    # The headline: how much of the apparent 90-degree-spaced variation is
    # present when nothing was resampled. A ratio near 1 means the 4-fold
    # signal is real; near 0 means it is an artefact of the interpolator.
    ratio = float(exact.mean() / interp.mean()) if np.isfinite(interp.mean()) \
        and interp.mean() > 0 else float("nan")
    # Paired per galaxy against that galaxy's own interpolated phases, which
    # removes galaxy-to-galaxy brightness and structure from the comparison.
    paired = np.stack([per_phase[p] for p in range(1, step)], axis=1).mean(axis=1)
    excess = interp.mean() - exact.mean()

    df["exact_varying_frac_mean"] = float(exact.mean())
    df["interpolated_varying_frac_mean"] = float(interp.mean())
    df["exact_over_interpolated_ratio"] = ratio
    df["resampling_excess_absolute"] = float(excess)
    df["resampling_share_of_interpolated"] = (
        float(excess / interp.mean()) if interp.mean() > 0 else np.nan)
    df["paired_exact_minus_interp_mean"] = float((exact - paired).mean())
    df["paired_exact_minus_interp_std"] = float((exact - paired).std(ddof=0))
    df["n_galaxies_favouring_exact_lower"] = int((exact < paired).sum())
    df["rotation_interpolation"] = a.rotation_interpolation
    df["embedding_preprocessing"] = a.embedding_preprocessing
    df["interpretation"] = (
        "phase 0 is resampling-free (np.rot90); phases 1+ use the declared "
        "interpolation kernel. All four sample 360 degrees at 90-degree "
        "spacing, so a ratio below 1 measures how much of the harmonic-4 "
        "signal the interpolator manufactured rather than the encoder saw.")

    print(f"  exact {exact.mean():.5f} vs interpolated {interp.mean():.5f} "
          f"varying-power fraction; ratio {ratio:.3f}; "
          f"{int((exact < paired).sum())}/{g} galaxies vary less when not resampled",
          file=sys.stderr)
    lc.write_table(a.out, df)


if __name__ == "__main__":
    sys.exit(main())
