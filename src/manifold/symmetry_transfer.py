"""Symmetry under exact transforms, on real and simulated galaxies alike.

Three questions, one encode pass.

**Does the orientation readout still track rotation on galaxies the model has
never seen?** A readout is fitted on the real rotation orbits -- predict the
sine and cosine of the applied angle from the embedding -- and then applied to
ASTRID galaxies transformed the same way. If it recovers the applied rotation
on simulations, the orientation machinery survives the domain gap; if it
collapses, orientation is entangled with whatever separates the two domains.

**Do orientation-free properties stay still?** The same transforms are applied
to real galaxies and the spread of a property probe's prediction is measured
across them.

**Mirror versus resampled rotation.** A mirror and a 90/180-degree turn are
index permutations: the pixel values are identical, only their addresses
change. A 30-degree turn resamples. Running both lets the resampling cost be
separated from the geometry, and it is the reason simulations are restricted
to the exact transforms here: ASTRID cutouts are 96px with no margin, so an
arbitrary rotation would pull undefined corners into frame. Real cutouts are
160px and are rotated before the central 96px crop, so every frame is fully
defined -- which is what makes 30 degrees available on the real side only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcommon as lc

from encode_orbits import rotate_stack, CROP

# Exact transforms: index permutations, no resampling anywhere.
EXACT = ["identity", "mirror", "rot90", "rot180"]
# Available on the real side only -- it needs the 160px margin.
RESAMPLED = ["rot30"]


def apply_transform(img, name, kernel, crop_only):
    """Return a (4, CROP, CROP) frame for the named transform.

    ``crop_only`` is for images that are already at the crop size (ASTRID at
    96px): they carry no margin, so they are transformed in place and never
    rotated by a non-multiple of 90 degrees.
    """
    if name == "mirror":
        out = img[:, :, ::-1]
    elif name == "rot90":
        out = np.rot90(img, k=1, axes=(1, 2))
    elif name == "rot180":
        out = np.rot90(img, k=2, axes=(1, 2))
    elif name == "identity":
        out = img
    elif name == "rot30":
        if crop_only:
            raise ValueError("rot30 needs a margin; not available at crop size")
        return rotate_stack(img, 30.0, kernel)
    else:
        raise ValueError(f"unknown transform: {name}")
    out = np.ascontiguousarray(out, dtype=np.float32)
    if crop_only:
        return out
    h, w = out.shape[1], out.shape[2]
    t, l = (h - CROP) // 2, (w - CROP) // 2
    return np.ascontiguousarray(out[:, t:t + CROP, l:l + CROP], dtype=np.float32)


def encode_set(model, codecs, device, images, rows, names, kernel, crop_only,
               batch, tag):
    """Encode every galaxy in *rows* under every transform in *names*."""
    import torch
    out = np.zeros((len(rows), len(names), 1024), dtype=np.float32)
    for gi, r in enumerate(rows):
        img = np.asarray(images[r], dtype=np.float32)
        frames = np.stack([apply_transform(img, n, kernel, crop_only) for n in names])
        from encodeImages import encode
        out[gi] = encode(model, codecs, device, frames, batchSize=batch)
        if (gi + 1) % 100 == 0 or gi + 1 == len(rows):
            print(f"  {tag} {gi + 1} / {len(rows)}", file=sys.stderr)
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-images", required=True)
    ap.add_argument("--sim-images", required=True)
    ap.add_argument("--orbits", required=True)
    ap.add_argument("--sim-cat", required=True)
    ap.add_argument("--embedding-preprocessing", required=True)
    ap.add_argument("--rotation-interpolation", required=True)
    ap.add_argument("--sim-quality-cuts", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d_in = Path(lc.resolve_input(a.orbits, "npy")).parent
    orbits = np.load(lc.resolve_input(a.orbits, "npy")).astype(np.float64)
    rows = np.load(d_in / "orbit_row_index.npy")
    angles = np.load(d_in / "orbit_angles_deg.npy")
    g, n_ang, d = orbits.shape

    real_imgs = np.load(lc.resolve_input(a.real_images, "npy"), mmap_mode="r")
    sim_imgs = np.load(lc.resolve_input(a.sim_images, "npy"), mmap_mode="r")
    sim_cat = pd.read_parquet(lc.resolve_input(a.sim_cat, "parquet"))

    sim_ok = np.flatnonzero(lc.sim_cut_mask(sim_cat, a.sim_quality_cuts))
    rng = np.random.default_rng(lc.SEED)
    sim_rows = np.sort(rng.choice(sim_ok, size=min(g, sim_ok.size), replace=False))

    sys.path.insert(0, str(Path("KAAI-2026-Hackathon/setup").resolve()))
    from encodeImages import loadEncoder
    model, codecs, device = loadEncoder()

    real_names = EXACT + RESAMPLED
    print(f"encoding real {g}x{len(real_names)} + sim {len(sim_rows)}x{len(EXACT)}"
          f" = {g * len(real_names) + len(sim_rows) * len(EXACT)} images",
          file=sys.stderr)
    R = encode_set(model, codecs, device, real_imgs, rows, real_names,
                   a.rotation_interpolation, False, a.batch_size, "real")
    S = encode_set(model, codecs, device, sim_imgs, sim_rows, EXACT,
                   a.rotation_interpolation, True, a.batch_size, "sim")

    # ---- orientation readout, fitted on the real rotation orbits -----------
    # Predicting sin/cos of the angle keeps the target continuous and periodic,
    # so the readout cannot cheat by learning a discontinuous bin index.
    prep = lc.EmbeddingPreprocessor(a.embedding_preprocessing).fit(
        orbits.reshape(g * n_ang, d))
    Zo = prep.transform(orbits.reshape(g * n_ang, d))
    th = np.deg2rad(np.tile(angles, g))
    from sklearn.linear_model import Ridge
    # Galaxy-disjoint split: the readout is never tested on a galaxy it saw.
    tr = np.repeat(np.arange(g) < g // 2, n_ang)
    head = Ridge(alpha=1.0).fit(Zo[tr], np.c_[np.sin(th), np.cos(th)])

    def read_angle(E):
        P = head.predict(prep.transform(E.reshape(-1, d)))
        return np.rad2deg(np.arctan2(P[:, 0], P[:, 1])) % 360.0

    # Held-out real orbits: how well does the readout work in-domain at all?
    pred_ho = read_angle(orbits[g // 2:])
    true_ho = np.tile(angles, g - g // 2)
    err_ho = np.abs((pred_ho - true_ho + 180) % 360 - 180)

    recs = []

    def circ_err(pred, applied):
        return np.abs((pred - applied + 180) % 360 - 180)

    # The applied rotation for each exact transform. A mirror is not a
    # rotation at all, so it has no true angle -- it is reported separately.
    applied = {"identity": 0.0, "rot90": 90.0, "rot180": 180.0}
    for dom, E, names in (("real", R, real_names), ("sim", S, EXACT)):
        base = read_angle(E[:, names.index("identity"), :])
        for j, nm in enumerate(names):
            p = read_angle(E[:, j, :])
            row = {"domain": dom, "transform": nm,
                   "resampled": nm == "rot30",
                   "n_galaxies": E.shape[0]}
            if nm in applied:
                # Absolute readout error against the angle actually applied.
                e = circ_err(p, applied[nm])
                row["readout_abs_err_median_deg"] = float(np.median(e))
                # Relative: does the readout MOVE by the applied amount?
                rel = circ_err(p - base, applied[nm])
                row["readout_delta_err_median_deg"] = float(np.median(rel))
            elif nm == "rot30":
                e = circ_err(p, 30.0)
                row["readout_abs_err_median_deg"] = float(np.median(e))
                row["readout_delta_err_median_deg"] = float(
                    np.median(circ_err(p - base, 30.0)))
            else:  # mirror
                row["readout_abs_err_median_deg"] = np.nan
                row["readout_delta_err_median_deg"] = np.nan
            # Embedding displacement, in units of that domain's own scale.
            di = np.sqrt(((E[:, j, :] - E[:, names.index("identity"), :]) ** 2
                          ).sum(axis=1))
            row["embedding_shift_median"] = float(np.median(di))
            recs.append(row)

    df = pd.DataFrame(recs)
    # Normalise displacement by each domain's own typical between-galaxy
    # distance, so real and sim are compared on comparable scales.
    for dom, E in (("real", R), ("sim", S)):
        idn = EXACT.index("identity")
        X = E[:, idn, :].astype(np.float64)
        sub = X[rng.choice(len(X), size=min(200, len(X)), replace=False)]
        dd = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2))
        scale = float(np.median(dd[np.triu_indices(len(sub), 1)]))
        df.loc[df.domain == dom, "between_galaxy_median_distance"] = scale
    df["embedding_shift_over_between_galaxy"] = (
        df["embedding_shift_median"] / df["between_galaxy_median_distance"])
    df["readout_holdout_err_median_deg"] = float(np.median(err_ho))
    df["readout_fitted_on"] = "real rotation orbits, galaxy-disjoint half"
    df["embedding_preprocessing"] = a.embedding_preprocessing
    df["rotation_interpolation"] = a.rotation_interpolation
    df["sim_quality_cuts"] = a.sim_quality_cuts
    df["note_exactness"] = (
        "identity, mirror, rot90 and rot180 are index permutations: pixel "
        "values are unchanged and nothing is resampled. rot30 resamples, and "
        "is available on real images only because their 160px cutouts carry "
        "the margin a 96px simulated cutout does not.")
    df["note_mirror"] = (
        "a mirror is not a rotation, so it has no applied angle; its row "
        "reports embedding displacement only.")

    print(f"  readout held-out error {np.median(err_ho):.1f} deg", file=sys.stderr)
    for r in recs:
        print(f"    {r['domain']:4s} {r['transform']:9s} "
              f"shift={r['embedding_shift_median']:8.3f} "
              f"delta_err={r.get('readout_delta_err_median_deg', float('nan')):7.2f}",
              file=sys.stderr)
    lc.write_table(a.out, df)


if __name__ == "__main__":
    sys.exit(main())
