"""Does the encoder see chirality, or only shape?

Rotation is SO(2); reflection is the other half of O(2), and nothing in the
AION-1 paper constrains either. The question matters because a mirror flip is
the one transformation that changes a spiral's winding direction while leaving
every other morphological property intact.

A raw similarity between a galaxy and its mirror answers nothing on its own,
because reflection and rotation are entangled: an ellipse or a bar is
two-fold symmetric, so flipping it is *the same thing* as rotating it, and a
large embedding shift would prove only that the encoder noticed the
orientation change. The sharp question is geometric:

    does the mirrored embedding land ON the galaxy's own rotation orbit?

If it does, reflection is absorbed by rotation and no handedness is encoded.
If it sits off the orbit, the encoder is carrying chirality — information a
rotation can never reproduce.

Two scales make that measurable. The orbit is sampled at a finite angle step,
so a point genuinely on the orbit still sits some distance from the nearest
*sampled* point; that grid spacing is the floor. And re-encoding each galaxy
unflipped reproduces its own stored angle-0 vector, which gives the numerical
noise floor of the pipeline itself. The mirror distance is reported against
both.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc

from encode_orbits import rotate_stack  # noqa: E402  (same crop/rotate recipe)


def _nearest_distance(z, P):
    """Distance from vector ``z`` to the nearest row of ``P``."""
    return float(np.sqrt(((P - z[None, :]) ** 2).sum(axis=1)).min())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--orbits", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d_in = Path(lc.resolve_input(a.orbits, "npy")).parent
    orbits = np.load(lc.resolve_input(a.orbits, "npy")).astype(np.float64)
    rows = np.load(d_in / "orbit_row_index.npy")
    angles = np.load(d_in / "orbit_angles_deg.npy")
    g, n_ang, d = orbits.shape
    if len(rows) != g:
        raise SystemExit(f"row index ({len(rows)}) does not match orbits ({g})")

    images = np.load(lc.resolve_input(a.images, "npy"), mmap_mode="r")

    sys.path.insert(0, str(Path("KAAI-2026-Hackathon/setup").resolve()))
    from encodeImages import loadEncoder, encode

    import torch
    model, codecs, device = loadEncoder()
    print(f"encoding {g} galaxies x 2 (identity + mirror) = {2 * g} images",
          file=sys.stderr)

    plain = np.zeros((g, d), dtype=np.float32)
    mirror = np.zeros((g, d), dtype=np.float32)
    for gi, row in enumerate(rows):
        img = np.asarray(images[row], dtype=np.float32)
        # Reflection about the vertical axis is an exact index reversal, so
        # like the 90-degree rotations it introduces no resampling of its own.
        flipped = img[:, :, ::-1].copy()
        frames = np.stack([rotate_stack(img, 0.0, "bicubic"),
                           rotate_stack(flipped, 0.0, "bicubic")])
        v = encode(model, codecs, device, frames, batchSize=a.batch_size)
        plain[gi], mirror[gi] = v[0], v[1]
        if (gi + 1) % 50 == 0 or gi + 1 == g:
            print(f"  galaxy {gi + 1} / {g}", file=sys.stderr)
        if device == "cuda":
            torch.cuda.empty_cache()

    # The metric is the one the orbit lives in: conditioning fitted on the
    # pooled orbit points, exactly as orbit_spectrum does, then applied to the
    # freshly encoded vectors.
    prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing)
    Z = prep.fit_transform(orbits.reshape(g * n_ang, d)).reshape(g, n_ang, d)
    Zp = prep.transform(plain.astype(np.float64))
    Zm = prep.transform(mirror.astype(np.float64))

    # The decisive null. "Close to the orbit" needs a reference measured the
    # same way, and the grid supplies one for free: split the angles into two
    # interleaved halves, treat the even ones as the grid and the odd ones as
    # points *known* to lie on the orbit, and measure odd-to-nearest-even. Any
    # genuine orbit point scores like that. The mirror is then read against it
    # rather than against an assumption about how a curve should be sampled.
    ev, od = np.arange(0, n_ang, 2), np.arange(1, n_ang, 2)
    null_ratio = np.full(g, np.nan)
    for i in range(g):
        G = Z[i][ev]
        on = np.sqrt(((Z[i][od][:, None, :] - G[None, :, :]) ** 2).sum(axis=2))
        dm = np.sqrt(((G[:, None, :] - G[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dm, np.inf)
        sp = dm.min(axis=1).mean()
        if sp > 0:
            null_ratio[i] = on.min(axis=1).mean() / sp

    recs = []
    for i in range(g):
        P = Z[i]
        # Floor 1: how far a genuine orbit point sits from its nearest sampled
        # neighbour -- the resolution of the angle grid.
        dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dm, np.inf)
        spacing = float(dm.min(axis=1).mean())
        # Floor 2: re-encoding the unflipped galaxy should reproduce its own
        # stored angle-0 vector. Whatever it does not reproduce is pipeline noise.
        noise = float(np.sqrt(((Zp[i] - P[0]) ** 2).sum()))
        d_mirror = _nearest_distance(Zm[i], P)
        radius = float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1)).mean())
        recs.append({
            "galaxy": int(rows[i]),
            "mirror_offorbit_distance": d_mirror,
            "orbit_grid_spacing": spacing,
            "reencode_noise": noise,
            "orbit_radius": radius,
            "mirror_over_spacing": d_mirror / spacing if spacing > 0 else np.nan,
            "mirror_over_noise": d_mirror / noise if noise > 0 else np.nan,
            "mirror_over_radius": d_mirror / radius if radius > 0 else np.nan,
            "nearest_orbit_angle_deg": float(angles[int(np.argmin(
                np.sqrt(((P - Zm[i][None, :]) ** 2).sum(axis=1))))]),
            "onorbit_null_ratio": float(null_ratio[i]),
        })

    df = pd.DataFrame(recs)
    med_ratio = float(df["mirror_over_spacing"].median())
    df["n_galaxies"] = g
    df["n_angles"] = n_ang
    df["embedding_preprocessing"] = a.embedding_preprocessing
    df["mirror_over_spacing_median"] = med_ratio
    df["mirror_over_spacing_mean"] = float(df["mirror_over_spacing"].mean())
    df["reencode_noise_median"] = float(df["reencode_noise"].median())
    df["orbit_grid_spacing_median"] = float(df["orbit_grid_spacing"].median())
    df["mirror_offorbit_distance_median"] = float(df["mirror_offorbit_distance"].median())
    df["frac_within_grid_spacing"] = float((df["mirror_over_spacing"] <= 1.0).mean())
    null_med = float(np.nanmedian(null_ratio))
    df["onorbit_null_ratio_median"] = null_med
    df["mirror_over_null"] = med_ratio / null_med if null_med > 0 else np.nan
    df["interpretation"] = (
        "onorbit_null_ratio is what a point known to be on the orbit scores: "
        "the distance from a held-out angle sample to the nearest sample of the "
        "interleaved half-grid, in units of that half-grid's spacing. "
        "mirror_over_spacing is the same quantity for the mirrored embedding. "
        "mirror_over_null near 1 means the mirror is indistinguishable from a "
        "genuine orbit point, so reflection is absorbed by rotation and no "
        "chirality is encoded; a value well above 1 means the mirror lies off "
        "the orbit and the encoder carries handedness. reencode_noise is the "
        "pipeline's own floor and is far below both.")

    print(f"  mirror {med_ratio:.3f} vs on-orbit null {null_med:.3f} "
          f"(ratio {med_ratio / null_med:.3f}); re-encode noise median "
          f"{df['reencode_noise_median'].iloc[0]:.4g}", file=sys.stderr)
    lc.write_table(a.out, df)


if __name__ == "__main__":
    sys.exit(main())
