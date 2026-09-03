"""Re-encode a subsample of anchor galaxies at a grid of position angles.

The only GPU-bound step in the project, and the measurement goal 2 exists
for: rotating a galaxy changes nothing physical about it, so whatever the
embedding does as the angle sweeps is the encoder's own orientation
degree of freedom, measured directly rather than assumed.

The recipe is the shipped one -- 600 encoder tokens, mean pooling, raw
nanomaggie flux, bands g,r,i,z -- because an embedding produced any other
way is not comparable to the ones the hackathon ships. The 160px anchor
cutouts are rotated first and cropped to the central 96px afterwards, so
every rotated frame is fully defined; this is exactly the margin the anchor
images were given for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc

SUBSAMPLE = {"n256": 256, "n512": 512, "n1024": 1024}
ANGLES = {"a8": 8, "a16": 16, "a32": 32}
CROP = 96


_LANCZOS_A = 3


def _lanczos_weights(t):
    """Separable Lanczos-3 taps for fractional offsets ``t``, rows normalised."""
    offs = np.arange(-_LANCZOS_A + 1, _LANCZOS_A + 1)          # 6 taps
    x = t[:, None] - offs[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.sinc(x) * np.sinc(x / _LANCZOS_A)
    w = np.where(np.abs(x) < _LANCZOS_A, w, 0.0)
    ssum = w.sum(axis=1, keepdims=True)
    return offs, np.divide(w, ssum, out=np.zeros_like(w), where=ssum != 0)


def _rotate_lanczos(img, angle_deg):
    """Rotate a (C, H, W) stack with a separable Lanczos-3 kernel.

    PIL refuses LANCZOS for rotation and scipy.ndimage offers only B-splines,
    so the sharpest of the three kernels is resampled here directly. The
    geometric convention is the same inverse map scipy.ndimage.rotate uses, so
    the three options differ only in the interpolation, which is the whole
    point of offering them.
    """
    c, h, w = img.shape
    t = np.radians(angle_deg)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r, k = np.meshgrid(np.arange(h, dtype=float), np.arange(w, dtype=float),
                       indexing="ij")
    dy, dx = r - cy, k - cx
    # Sign convention matched empirically to scipy.ndimage.rotate, so the
    # three kernels describe the same rotation and differ only in resampling.
    sy = cy + np.cos(t) * dy + np.sin(t) * dx
    sx = cx - np.sin(t) * dy + np.cos(t) * dx

    fy, fx = np.floor(sy).astype(int), np.floor(sx).astype(int)
    offs, wy = _lanczos_weights((sy - fy).ravel())
    _, wx = _lanczos_weights((sx - fx).ravel())
    fy, fx = fy.ravel(), fx.ravel()

    out = np.zeros((c, fy.size), dtype=np.float64)
    for a, oy in enumerate(offs):
        yy = fy + oy
        vy = (yy >= 0) & (yy < h)
        yc = np.clip(yy, 0, h - 1)
        for b, ox in enumerate(offs):
            xx = fx + ox
            vx = (xx >= 0) & (xx < w)
            xc = np.clip(xx, 0, w - 1)
            ww = wy[:, a] * wx[:, b] * (vy & vx)
            out += img[:, yc, xc] * ww
    return out.reshape(c, h, w).astype(np.float32)


def rotate_stack(img, angle_deg, kernel):
    """Rotate a (4, H, W) cutout about its centre and crop the central 96px.

    Rotation resamples, and resampling smooths -- a change the encoder can
    see. Which kernel does the smoothing is therefore itself a decision, and
    the exact 90-degree multiples are handled without resampling at all so
    they stand as the interpolation-free control.
    """
    if angle_deg % 90 == 0:
        out = np.rot90(img, k=int(angle_deg // 90) % 4, axes=(1, 2))
    elif kernel == "lanczos":
        out = _rotate_lanczos(img, angle_deg)
    else:
        from scipy.ndimage import rotate
        order = {"bilinear": 1, "bicubic": 3}[kernel]
        out = rotate(img, angle_deg, axes=(1, 2), reshape=False, order=order,
                     mode="constant", cval=0.0, prefilter=(order > 1))
    h, w = out.shape[1], out.shape[2]
    t, l = (h - CROP) // 2, (w - CROP) // 2
    return np.ascontiguousarray(out[:, t:t + CROP, l:l + CROP], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--cat", required=True)
    ap.add_argument("--real-quality-cuts", required=True)
    ap.add_argument("--orbit-subsample", required=True)
    ap.add_argument("--angle-grid", required=True)
    ap.add_argument("--rotation-interpolation", required=True)
    ap.add_argument("--orbit-reference-frame", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.orbit_subsample not in SUBSAMPLE:
        raise ValueError(f"unknown orbit_subsample: {a.orbit_subsample}")
    if a.angle_grid not in ANGLES:
        raise ValueError(f"unknown angle_grid: {a.angle_grid}")
    n_gal, n_ang = SUBSAMPLE[a.orbit_subsample], ANGLES[a.angle_grid]

    cat = pd.read_parquet(lc.resolve_input(a.cat, "parquet"))
    # 4 GB of cutouts: memory-mapped so only the sampled rows are ever read.
    images = np.load(lc.resolve_input(a.images, "npy"), mmap_mode="r")
    if len(cat) != len(images):
        raise ValueError(f"row misalignment: {len(images)} images vs {len(cat)} rows")

    eligible = np.flatnonzero(lc.real_cut_mask(cat, a.real_quality_cuts))
    rng = np.random.default_rng(lc.SEED)
    sel = np.sort(rng.choice(eligible, size=min(n_gal, eligible.size), replace=False))
    angles = np.arange(n_ang) * (360.0 / n_ang)

    if a.orbit_reference_frame == "canonical_pa":
        # De-rotate each galaxy so its major axis is horizontal before the
        # grid is applied: orbits then share a phase origin by construction,
        # at the cost of depending on the catalogue's position angles.
        pa = pd.to_numeric(cat["paDeg"], errors="coerce").to_numpy(dtype=float)[sel]
        pa = np.where(np.isfinite(pa), pa, 0.0)
    elif a.orbit_reference_frame == "sky_frame":
        pa = np.zeros(sel.size)
    else:
        raise ValueError(f"unknown orbit_reference_frame: {a.orbit_reference_frame}")

    sys.path.insert(0, str(Path("KAAI-2026-Hackathon/setup").resolve()))
    from encodeImages import loadEncoder, encode

    import torch
    model, codecs, device = loadEncoder()
    print(f"encoding {sel.size} galaxies x {n_ang} angles = {sel.size * n_ang} images",
          file=sys.stderr)

    out = np.zeros((sel.size, n_ang, 1024), dtype=np.float32)
    for gi, row in enumerate(sel):
        img = np.asarray(images[row], dtype=np.float32)
        frames = np.stack([rotate_stack(img, float(angles[k] - pa[gi]),
                                        a.rotation_interpolation)
                           for k in range(n_ang)])
        out[gi] = encode(model, codecs, device, frames, batchSize=a.batch_size)
        if (gi + 1) % 25 == 0 or gi + 1 == sel.size:
            print(f"  galaxy {gi + 1} / {sel.size}", file=sys.stderr)
        if device == "cuda":
            torch.cuda.empty_cache()

    lc.write_array(a.out, out)
    # The row indices are what ties an orbit back to its catalogue row; every
    # downstream output needs them, so they travel with the array.
    np.save(Path(a.out) / "orbit_row_index.npy", sel)
    np.save(Path(a.out) / "orbit_angles_deg.npy", angles)
    print(f"wrote row index ({sel.size}) and angle grid ({n_ang})", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
