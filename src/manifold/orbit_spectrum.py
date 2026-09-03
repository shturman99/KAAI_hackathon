"""Angular harmonic content of the rotation orbit.

Rotating a galaxy by a full turn returns it to itself, so an orbit is a
closed, periodic curve and its natural description is a Fourier series in the
angle. The question this answers is whether the orbit is essentially a circle
-- power concentrated in harmonic 1 once the constant term is set aside -- or
carries richer angular structure. An N-point grid resolves harmonics up to
N/2 before aliasing, which is what the angle_grid decision buys.
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
    ap.add_argument("--orbits", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    orbits = np.load(lc.resolve_input(a.orbits, "npy")).astype(np.float64)
    g, n_ang, d = orbits.shape

    # Conditioning is fitted on the pooled orbit points: the spectrum is a
    # statement about variation across angle, so every angle must be measured
    # on the same scale.
    prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing)
    Z = prep.fit_transform(orbits.reshape(g * n_ang, d)).reshape(g, n_ang, d)

    F = np.fft.rfft(Z, axis=1) / n_ang
    power = np.abs(F) ** 2                       # (g, n_harmonics, d)
    per_harmonic = power.sum(axis=2)             # summed over embedding dims
    n_h = per_harmonic.shape[1]

    # Harmonic 0 is the orbit mean -- the rotation-invariant part -- so the
    # fraction that matters is taken over the varying harmonics only.
    varying = per_harmonic[:, 1:]
    tot = varying.sum(axis=1, keepdims=True)
    frac = np.divide(varying, tot, out=np.zeros_like(varying), where=tot > 0)

    rows = []
    for h in range(n_h):
        p = per_harmonic[:, h]
        row = {
            "harmonic": h,
            "angular_period_deg": 360.0 / h if h else np.inf,
            "power_mean": float(p.mean()),
            "power_std": float(p.std(ddof=0)),
            "power_median": float(np.median(p)),
            "is_rotation_invariant_part": bool(h == 0),
            "frac_of_varying_power_mean": (float(frac[:, h - 1].mean()) if h else np.nan),
            "frac_of_varying_power_std": (float(frac[:, h - 1].std(ddof=0)) if h else np.nan),
            "n_galaxies": int(g),
            "n_angles": int(n_ang),
            "embedding_preprocessing": a.embedding_preprocessing,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    dom = int(np.argmax(df["frac_of_varying_power_mean"].to_numpy()[1:]) + 1)
    # A circle would put essentially all varying power in a single harmonic;
    # how concentrated it actually is decides whether "the orbit is a circle"
    # survives contact with the measurement.
    df["dominant_varying_harmonic"] = dom
    df["dominant_harmonic_power_frac"] = float(df.loc[dom, "frac_of_varying_power_mean"])
    df["invariant_power_frac_mean"] = float(
        (per_harmonic[:, 0] / per_harmonic.sum(axis=1)).mean())

    print(f"  dominant varying harmonic {dom} carrying "
          f"{df.loc[dom, 'frac_of_varying_power_mean']:.3f} of varying power; "
          f"harmonic 0 holds {df['invariant_power_frac_mean'].iloc[0]:.4f} of total",
          file=sys.stderr)
    lc.write_table(a.out, df)


if __name__ == "__main__":
    sys.exit(main())
