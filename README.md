# Symmetry in a frozen astronomy foundation model

**Team: claude maxing** — KAAI 2026 Hackathon, Carnegie Mellon, 3 September 2026

We answer the **Symmetry** question: when a galaxy image is rotated or mirrored,
what does AION-1-Large's frozen image embedding do about it?

Every number below is produced by a script in this repository and recorded with
its full provenance. Nothing is typed by hand.

---

## The question, and what we found

> *When a picture is rotated or mirrored, do properties with no orientation stay
> still? Does the orientation readout still track rotation on simulated galaxies
> the model has never seen? A mirror moves no pixels. A rotation by 30 degrees
> resamples them. Compare the two.*

| # | Finding | Number |
|---|---------|--------|
| 1 | The encoder is **partially rotation-invariant, though nobody trained it to be** — the AION-1 paper never mentions rotation, flip or augmentation | 80.9% of embedding power is rotation-invariant |
| 2 | Of the part that moves, **m=2 (180°) dominates** — the signature of a two-fold symmetric shape (disc ellipse or bar) | 37.7% of varying power |
| 3 | **That signal is real, not a resampling artefact.** m=4 sits at 90°, the pixel grid's own symmetry, so we tested it against interpolation-free rotations | ratio 0.996; resampling = 0.45% |
| 4 | **Chirality is not encoded.** A mirrored galaxy lands on its own rotation orbit, so the model reads spiral arms but not their winding direction | 0.973 of the on-orbit null |
| 5 | **Orientation costs ~2.4 of ~14 intrinsic dimensions** — not the 4.3 a naive comparison gives, which is half estimator bias | 2.37 ± 0.60 at matched sample size |

Supporting context (same codebase, not the symmetry question itself): the
embedding beats catalogue photometry on morphology (+0.19 to +0.30 AUC) but only
marginally on physical parameters (+0.06 to +0.13 R²); and real vs simulated
galaxies are perfectly separable (AUC 1.000) at every control stage, which is
**confound-limited** — ASTRID images have no PSF and no noise, so that number is
an upper bound on the physics gap, not a measurement of it.

### Why the experiment is designed the way it is

Two design choices carry most of the weight:

**Exact transforms remove the resampling confound by construction.** A mirror,
and rotations by 90°/180°, are *index permutations*: the pixel values are
identical, only their addresses change. A 30° rotation resamples and smooths,
which the encoder can see. Our `encode_orbits.py` routes exact 90° multiples
through `np.rot90` and everything else through the interpolation kernel — so the
16-angle grid already contains one interpolation-free sub-orbit and three
resampled ones at identical geometry. Comparing them isolates what interpolation
manufactures. **This cost zero GPU time**: the control reuses embeddings already
on disk.

**Real images are 160 px, simulated are 96 px.** Anchor cutouts are rotated
*then* cropped to the central 96 px, so every rotated frame is fully defined.
ASTRID cutouts are already 96 px with no margin, so only exact transforms
(mirror, 90°, 180°) can be applied without pulling undefined corners into frame.
That is why the cross-domain test uses exact transforms only — which also means
any real-vs-simulated difference there **cannot** be blamed on resampling.

### What we did not get to

Two scripts are in the repository but **were not run**, and no number in this
README or the report depends on them:

- `symmetry_transfer.py` — the cross-domain test (does the orientation readout
  still track rotation on ASTRID?). It is written and wired into `astra.yaml`;
  the 6 GB laptop GPU thermally throttled to 352 MHz at 89 °C and it would not
  have finished before the deadline.
- `property_invariance.py` — whether orientation-free properties stay still
  across the orbit. Blocked on the same ~30 min morphology probe cost.

Had `symmetry_transfer` run, the four-causes attribution would have been: for the
exact transforms resampling is eliminated by construction (identical pixel
values), leaving different picture-making (no PSF, no noise in ASTRID) as the
leading candidate — the same confound that saturates the domain-gap AUC at 1.000.
Different galaxies and genuine physics cannot be separated from it without
PSF-matching the simulations and re-encoding.

---

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned to the hackathon's versions
pip install lightcone-cli                # orchestration

lc run                                   # materialize everything
lc run manifold.parity_control           # or one output
lc status                                # what is materialized
lc verify                                # re-hash every artifact
```

Outputs land in `analyses/<sub-analysis>/results/<universe>/<output_id>/` with a
`.lightcone-manifest.json` recording the recipe, decisions, input hashes and
output hash.

**Requirements.** The GPU steps need CUDA and ~5.4 GB of VRAM. Everything else is
CPU-only. `KAAI-2026-Hackathon/` is the provided data kit and is read-only input.

**Two practical notes if you re-run this:**

- `lc status` reports every output `stale` on this project even right after a
  successful run. This is a false alarm: the `Containerfile` ends in `COPY . .`,
  so the image digest feeding `code_version` hashes the project root — including
  the run logs each run writes on its way out, so every run invalidates its own
  hash. **`lc verify` is the authority** — it re-hashes the actual artifacts.
- A bare `lc run <output>` fans out across *all* universes. Pass `-u baseline`
  unless you mean to rebuild the sensitivity universes too (one of them re-runs
  the 20-minute GPU encode).

---

## Which script makes which number

| Script | Output | What it produces |
|---|---|---|
| **The symmetry answer** | | |
| `src/manifold/encode_orbits.py` | `manifold.rotation_orbit_embeddings` | 512 galaxies × 16 angles = 8,192 encodes. **GPU, ~20 min.** Routes exact 90° multiples through `np.rot90` (no resampling) |
| `src/manifold/orbit_spectrum.py` | `manifold.orbit_fourier_spectrum` | Angular harmonic spectrum → **80.9% invariant, m=2 at 37.7%, m=4 at 17.9%** |
| `src/manifold/exact_rotation_control.py` | `manifold.exact_rotation_control` | Interpolation-free vs resampled sub-orbits → **ratio 0.996** (finding 3). No GPU |
| `src/manifold/parity_control.py` | `manifold.parity_control` | Mirror vs its own rotation orbit → **0.947 vs 0.974 null** (finding 4). **GPU, ~6 min** |
| `src/manifold/symmetry_transfer.py` | `manifold.symmetry_transfer` | **Written, not run.** Orientation readout on real *and* ASTRID under mirror/90/180, plus 30° on real. Abandoned: the 6 GB laptop GPU throttled to 352 MHz at 89 °C |
| `src/manifold/property_invariance.py` | `manifold.property_invariance` | **Written, not run.** Do orientation-free properties stay still across the orbit? Needs ~30 min for the morphology probes (60 logistic solves per fit) |
| **Manifold geometry** | | |
| `src/manifold/intrinsic_dim.py` | `intrinsic_dim_full` / `_orbit` / `_quotient` / `_matched` | TwoNN dimensionality. `--mode matched` gives the **2.37 ± 0.60** sample-size-corrected figure (finding 5) |
| `src/manifold/plot_dimensionality.py` | `manifold.id_comparison_plot` | Dimensionality budget figure |
| `src/manifold/symbolic_regression.py` | `manifold.symbolic_expressions` | Genetic symbolic search on quotiented coordinates |
| `src/manifold/symbolic_complexity_curve.py` | `manifold.symbolic_complexity_curve` | Held-out R² at every complexity |
| `src/manifold/plot_complexity_curve.py` | `manifold.symbolic_complexity_plot` | Accuracy-vs-complexity figure |
| `src/manifold/symbolic_vs_linear.py` | `manifold.symbolic_vs_linear_r2` | Symbolic vs linear probe |
| **What is decodable (context)** | | |
| `src/probes/probe_physical.py` | `probes.probe_r2_physical` | Embedding R² on 5 physical targets |
| `src/probes/probe_photometry.py` | `probes.probe_r2_photometry_baseline` | The photometry bar those R² are read against |
| `src/probes/probe_morphology.py` | `probes.morphology_auc` | Morphology AUC from embeddings. **Slow (~30 min)**: 60 logistic solves per fit |
| `src/probes/probe_morphology_photometry.py` | `probes.morphology_auc_photometry` | Morphology AUC from colour alone |
| `src/probes/probe_nuisance.py` | `probes.nuisance_probe_r2` | 17 observing-condition probes (R² 0.28–0.73) **+ the weight vectors** `domain_gap` consumes |
| `src/probes/plot_comparison.py` | `probes.probe_comparison_plot` | Embedding-vs-photometry figure |
| **Real vs simulated (context)** | | |
| `src/domain_gap/discriminate.py` | `raw_` / `matched_` / `residual_discriminability` | Three-stage AUC staircase, all 1.000 |
| `src/domain_gap/probe_transfer.py` | `domain_gap.probe_transfer_r2` | Train-on-real, test-on-sim transfer |
| `src/domain_gap/plot_gap.py` | `domain_gap.domain_gap_plot` | Domain-gap figure |

`src/lcommon.py` holds every decision's implementation exactly once, so two
outputs declaring the same decision cannot drift apart.

---

## Repository layout

```
astra.yaml                  the specification: inputs, decisions, outputs
analyses/<name>/astra.yaml  the three sub-analyses
universes/                  decision settings; baseline plus sensitivity sweeps
src/                        one module per output, plus src/lcommon.py
analyses/*/results/         materialized outputs + provenance manifests
```

**Universes.** `baseline` is the headline configuration. `nuis_d5` / `nuis_d20`
sweep the nuisance-subspace dimension (the residual AUC is 1.000 at all of them).
`fixedpen` holds the probe penalty fixed instead of tuning it, used for the
transform-stability measurements where the question is how much a fitted readout
*moves*, not how accurate it is.

## Honest limitations

- **The real/simulated result is confound-limited.** ASTRID has no PSF and no
  noise; we control only the *linearly* captured instrument subspace. AUC 1.000
  is an upper bound on the physics gap.
- **16 angles resolves harmonics only to m=8** (Nyquist). m=4 already carries
  17.9%, so power above m=8 could alias down. A 32-angle run was attempted; the
  6 GB laptop GPU thermally throttled to 270 MHz at 88 °C and it was abandoned.
- **Chirality was tested geometrically, not against labels.** The orbit galaxies
  come from the anchor sample, whose morphology columns are 1.9–3.2% covered;
  the labelled `gzArm` sample is a *different* set of galaxies and the two are
  never merged.
- **512 galaxies, not 10,000**, for everything requiring re-encoding — a 6 GB
  VRAM limit. This is what forced the sample-size correction in finding 5.

## Team

**claude maxing**

| | |
|---|---|
| Jack O'Brien | jackob@illinois.edu |
| Murman Grugenidze | mgurgeni@andrew.cmu.edu |

## Data

`KAAI-2026-Hackathon/` is the provided data kit (9.2 GB) and is **not** in this
repository. Download it from the organisers and place it at the repository root;
every input path in `astra.yaml` resolves relative to it.
