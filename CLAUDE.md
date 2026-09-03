# Project Notes for Claude

This is an ASTRA project orchestrated by `lightcone-cli`. It has been
scoped: `astra.yaml` holds the real specification — three sub-analyses,
19 outputs, 29 decisions, and 29 verified literature quotes. No
implementation code exists yet; `src/` is empty and every recipe names a
script that still has to be written.

Once scoped, the `lc` CLI keeps the substrate in sync:

```
lc run                    # all outputs in the default universe
lc run output_id          # one specific output
lc status                 # show what's materialized vs stale vs missing
lc verify                 # validate the provenance chain
```

Outputs land in `results/<universe>/<output_id>/` along with a sidecar
`.lightcone-manifest.json` recording the recipe, container, decisions,
input hashes, and output hash.

## Report

`index.md` + `myst.yml` are a template MyST report wired to the MySTRA
plugin. The report references analysis elements by path — inline mentions
with the `{astra}` role, block embeds with the `{astra}` directive, live
numbers with `{astra:value}` — so never hard-type a measured value in the
prose. Preview with `myst start` (requires the MyST CLI, `npm i -g mystmd`).


## Project Notes

### What this is

Built for the KAAI 2026 Hackathon (Carnegie Mellon, 2–3 September 2026),
whose brief is to pose an interesting research question against a frozen
astronomy foundation model. The kit lives in `KAAI-2026-Hackathon/` and is
read-only input: 10,000 real galaxies with a 190-column catalogue
(`observational-data/anchor.parquet`), a separate 10,000-galaxy morphology
sample (`gzArm.parquet`), and 10,000 simulated ASTRID galaxies. All
embeddings are precomputed with AION-1-Large, mean-pooled at a 600-token
budget. `setup/encodeImages.py` reproduces that exact recipe for images you
encode yourself — match it or the vectors are not comparable.

### Three goals, three sub-analyses

1. `probes` — a deliberately linear baseline. What is decodable at all, and
   how much of it is already in catalogue photometry.
2. `manifold` — the goal the project exists for. Re-encode galaxies at many
   rotation angles, measure the orbit, quotient orientation out, and ask
   what dimensionality and structure remain. Then symbolic regression.
3. `domain_gap` — real vs simulated, using features both domains share.

### Facts established during scoping that are not derivable from the data

- **The GPU is an RTX 3060 laptop card, 6 GB.** AION-1-Large (~0.9B params)
  fits in fp16 for inference. `manifold.rotation_orbit_embeddings` is the
  only GPU-bound output and is deliberately subsampled (512 galaxies × 16
  angles = 8,192 encodes) to finish on it. This is why `orbit_subsample`
  exists as a decision.
- **Image degradation was ruled out of scope.** The honest way to compare
  real and simulated images would be to convolve the sims with a matched PSF,
  inject noise from `anchorIvar`, and re-encode. That was considered and
  declined on time grounds. `domain_gap` instead controls the confound by
  distribution matching plus nuisance-subspace projection. This is the
  central limitation of goal 3 and must be stated wherever its numbers are
  reported.
- **Morphology cannot come from the anchor sample.** Its morphology columns
  are 1.9–3.2% covered and `morphReliable` is true for only 33%. That is why
  `probes` reads morphology exclusively from `gzArm`. The two catalogues are
  different galaxies and must never be merged.
- **gzArm labels are severely imbalanced**: `bulgeDominantFraction` median
  0.011, `mergerFraction` 0.022, `spiralArmsFraction` 0.88. A 0.5 threshold
  produces meaningless AUCs, which is what `morphology_label_form` (default
  `threshold_median`) exists to handle.
- **`paDeg` is 100% covered** in both real catalogues. Currently unused —
  `orbit_reference_frame` defaults to `sky_frame` deliberately, to keep the
  rotation experiment independent of catalogue position angles. The
  `canonical_pa` option is the alternative if orbit stacking proves messy.

### What the literature settled, and what it did not

Three papers extracted, 29 quotes verified against cached PDFs
(`astra validate astra.yaml --verify-evidence`). Note the extractors found
the default `astra paper add` cache resolves to arXiv **v2** for some DOIs;
all evidence here pins `version: 1` and verifies against the `_v1` cache
entries.

- **AION-1** (`10.48550/arXiv.2510.17960`) — justified mean pooling
  (attentive pooling gave no gain on image tasks); supplied the 80/20 and
  light-head protocol; and supplied the sSFR convention that changed
  `target_transform`'s default to `log_ssfr`. Two negatives matter as much:
  the paper reports **no image-only configuration for its own model**, so
  image-only yardsticks come from its AstroCLIP/DINOv2 baseline rows
  (Table 1, p.14 — tabular, so not verbatim-quotable; read it directly
  before citing a number); and it contains **no mention of rotation, flip,
  or augmentation anywhere**, so nothing constrains whether these embeddings
  are rotation-equivariant. Goal 2 is genuinely open ground.
- **Ansuini et al. 2019** (`10.48550/arXiv.1905.12784`) — fixed
  `id_estimator` on TwoNN and supplied the caveat that above ID ≈ 20 it
  underestimates, so a large full-cloud value is a **lower bound**, not an
  estimate. Also supplied the stability diagnostic behind
  `id_scale_selection: block_analysis`. Benchmark: last-layer IDs of trained
  image CNNs cluster around 12–25 despite 512–4096 units.
- **Cranmer et al. 2020** (`10.48550/arXiv.2006.11287`) — established that
  compression before symbolic regression is mandatory, not optional; the
  Pareto selection rule behind `sr_complexity_selection: best_score`; the
  constant-refit control behind the added `sr_validation` decision; and the
  warning that a PCA basis is arbitrary up to rotation, which is why
  `sr_reduction` carries `sparse_pca` and `probe_orthogonal` alternatives.

### Coupling to preserve

`domain_gap` takes `probes.nuisance_probe_r2` as a sibling input. That file
must carry the fitted probe **weight vectors**, not just R² values — the
weights span the instrument subspace that `residual_discriminability`
projects out. Nothing else in the project depends on cross-sub-analysis
wiring.

### Decisions deliberately left at defensible-but-arbitrary defaults

`nuisance_subspace_dim` (d10) and `sr_n_coordinates` (k8) are both round
numbers with no evidence behind them. `sr_n_coordinates` should be revisited
against the measured `intrinsic_dim_quotient` rather than left fixed. If
`residual_discriminability` disagrees between d5 and d20, report it as a
function of the dimension rather than at a single value.

## Implementation notes

`src/` is written. One module per output, plus `src/lcommon.py`, which holds
every decision's implementation exactly once so that two outputs declaring
the same decision cannot drift apart. Scripts add `src/` to `sys.path` and
`import lcommon as lc`; they are run only through `lc run`.

### Layout change made during implementation

The spec originally declared the three sub-analyses **inline** under
`analyses:`. `astra validate` accepts that, but lightcone-cli 0.4.2 only
merges a sub-analysis's local decisions into a universe when the sub-analysis
is **path-rooted** (`engine/tree.py`: `sub_path = analysis_node.get("path")`
followed by `if not sub_path: continue`), so every `{decisions.<local_id>}`
placeholder failed to render. The three sub-analyses now live in
`analyses/<name>/astra.yaml` with their own `universes/baseline.yaml`; the
root spec references them by `path:`. Ids, decisions, options, insights and
recipes were carried across unchanged — a YAML round-trip confirms the split
is content-identical to the original apart from the three edits below.
Results now land under `analyses/<name>/results/<universe>/<output_id>/`.

### Spec edits made to honour declared decisions

Three outputs declared a quality-cut decision but were not given the
catalogue the cut is applied to, so the decision could not have taken effect.
Each gained the catalogue it needed as an input and a matching recipe flag:

- `manifold.intrinsic_dim_full` — `anchor_catalogue`
- `manifold.intrinsic_dim_quotient` — `anchor_catalogue`
- `domain_gap.raw_discriminability` — `anchor_catalogue`, `astrid_catalogue`

### Environment facts

- **Docker is not usable on this machine** (permission denied on
  `/var/run/docker.sock`), so `~/.lightcone/config.yaml` is set to
  `container.runtime: none` and recipes run natively in `.venv`. Restore it
  to `auto` once the account is in the `docker` group.
- The venv is pinned to the hackathon's own versions (`requirements.txt`)
  because `rotation_orbit_embeddings` re-encodes images with the same frozen
  model and the vectors must stay comparable. `safetensors` is an extra pin:
  `huggingface_hub` 1.7.2 dereferences `safetensors.__version__`
  unconditionally in `from_pretrained` and raises `NameError` without it.
- **The encode recipe was verified against the shipped embeddings**: running
  the unrotated central 96px crop through `setup/encodeImages.py` reproduces
  `anchorEImg.npy` to cosine 0.999998. Rotation-orbit vectors are therefore
  comparable to every other embedding in the project.
- Measured cost of the GPU step: ~137 ms/image at batch 16, so 512 x 16 =
  8,192 encodes takes ~20 minutes and peaks at ~5.4 GB of the card's 6 GB.
  Batch 16 is the natural size — one galaxy is one batch.
- The box has 20 cores but only ~4 GB of free RAM. `n_jobs=-1` inside
  `LogisticRegressionCV` OOMs it, and unbounded BLAS fan-out costs more in
  contention than it returns, so `lcommon` caps thread counts at 8 on import.

### Implementation choices the spec left open

- **`cv_grid` uses `SvdRidgeCV`, not `sklearn.RidgeCV(cv=5)`.** The penalty
  enters ridge only through the eigenvalues of `X'X`, so one eigendecomposition
  per fold serves the whole alpha grid instead of one refit per (alpha, fold)
  pair. Verified to agree with sklearn to 1e-15 on a matched splitter; it is
  ~20x faster, which is what makes a 28-point grid at 1024 features affordable.
- **`ridge` maps to an L2-penalised logistic classifier** wherever the target
  is binary, so one decision names one model family across regression and
  classification outputs.
- **The discriminator's penalty is fixed at C=1.0.** The three
  `*_discriminability` outputs declare `classifier_model` but not
  `probe_regularization`, so the penalty is not a decision there; it is held
  fixed and recorded in the artifact rather than tuned invisibly.
- **Symbolic regression is a local genetic search** (`src/manifold/gp_symbolic.py`),
  not PySR, which would need a Julia toolchain. Written so that every knob the
  spec names is a real knob: bounded operator set, node-count complexity,
  Pareto front, separable constant refit, and the pre-registered selection rule.
- **Lanczos rotation is implemented directly.** PIL refuses `LANCZOS` for
  rotation and `scipy.ndimage` offers only B-splines. The separable Lanczos-3
  gather in `encode_orbits.py` matches `scipy.ndimage.rotate`'s geometry to a
  correlation of 0.99996, so the three kernels differ only in resampling.
- **Where the reduction is fitted vs applied.** For `orbit_mean` and
  `pca_deflation` the quotient lives in the shipped embedding's ambient space,
  so the SR reduction is *fitted* on the 512 quotiented orbits and *applied* to
  the full anchor cloud — which is why `symbolic_expressions` and
  `intrinsic_dim_quotient` both declare `anchor_embeddings` as an input.
  `harmonic_magnitudes` changes the ambient dimension, so there the fit is
  restricted to the orbit subsample and the artifact says so.
- **`intrinsic_dim_orbit` applies the stability rule over galaxies.** A single
  orbit has only `n_angles` points, far too few to block-analyse. The per-galaxy
  ID is estimated over its own angle points and the block analysis then runs
  over the *population* of galaxies, which is the sampling question the small
  `orbit_subsample` actually raises.
- **Two options are refused rather than faked**, with an explicit error:
  `domain_matching: mass_redshift_size` (ASTRID ships no apparent-size column;
  `enclR` is enclosed-light fraction, not size) and
  `nuisance_projection: cca_subspace` (needs the nuisance variables themselves,
  which `nuisance_probe_r2` does not carry).

### The cross-sub-analysis wire

`probes.nuisance_probe_r2` writes one row per nuisance variable with `r2_mean`,
`r2_std` and 1024 weight columns `w0000..w1023`. The weights are stored as a
linear functional on the **raw** embedding and unit-normalised, so a consumer
conditioning the embedding differently can re-express them
(`EmbeddingPreprocessor.direction_to_transformed`) and every nuisance variable
counts equally when the subspace is formed. `domain_gap` re-expresses them in
each fold's conditioned space, takes an SVD, and keeps the top
`nuisance_subspace_dim` directions — capped at the rank actually available and
recorded when capped.
