# What does a frozen astronomy foundation model encode in a galaxy image?

AION-1-Large was trained on real observations and then frozen. This report
takes its image-only embeddings for 10,000 real galaxies, 10,000
morphology-labelled galaxies and 10,000 simulated ASTRID galaxies, and asks
three questions of them: what is linearly decodable, what shape the embedding
cloud is once orientation is quotiented out, and how much of the model's
ability to tell a real galaxy from a simulated one is physics rather than the
absence of a point spread function.

Every number below is pulled live from the materialized analysis. None is
typed by hand.

## Methods

The embedding is conditioned before anything else is measured, because the
choice changes both the probe fits and the metric the nearest-neighbour
dimension estimators work in:

:::{astra} decisions.embedding_preprocessing
:::

Every reported score is held out under
{astra}`decisions.eval_split`, targets are transformed under
{astra}`decisions.target_transform`, and the probe family is
{astra}`decisions.probe_model`.

## Goal 1 — what is linearly decodable

The measurement is only meaningful against a bar. `probes` fits the same
targets from catalogue photometry alone and reports both, so the claim is
about what the pixels add rather than about a number being large.

:::{astra} analyses.probes.outputs.probe_r2_physical
:::

:::{astra} analyses.probes.outputs.probe_comparison_plot
:::

The nuisance probes in the right-hand panel are a warning rather than a
result: they measure how much of the embedding is telescope instead of
galaxy. Their fitted directions are what goal 3 projects out.

## Goal 2 — the shape of the cloud

Each of a subsample of galaxies is re-encoded at a grid of position angles,
so its rotation orbit is measured directly rather than assumed. The encoder
recipe is the shipped one, which is what keeps these vectors comparable to
every other embedding here.

:::{astra} analyses.manifold.outputs.id_comparison_plot
:::

Read the middle bar with its flag attached. A rotation orbit is sampled on a
uniform angle grid by construction, which violates TwoNN's local-Poisson
assumption, so the harmonic spectrum and the linear participation ratio carry
the answer there instead.

:::{astra} analyses.manifold.outputs.symbolic_expressions
:::

The coordinates these expressions are written in come from a PCA of the
quotiented cloud, and a PCA basis is arbitrary up to rotation — so an
expression is a statement about the manifold, not about a named embedding
direction.

## Goal 3 — real versus simulated

ASTRID images carry no PSF and no noise, so a naive classifier here is a
noise detector. Three stages of control say how much survives:

:::{astra} analyses.domain_gap.outputs.domain_gap_plot
:::

:::{astra} decisions.nuisance_projection
:::

**The central limitation of this goal.** The honest control would convolve
the simulated images with a matched PSF, inject noise from `anchorIvar` and
re-encode. That was considered during scoping and declined on time grounds.
What is done instead is distribution matching plus projection of the
instrument subspace the nuisance probes capture **linearly**. Instrument
signature that is encoded nonlinearly survives that projection, so the
residual number is an upper bound on the physics gap, not a measurement of
it. This applies wherever
{astra}`analyses.domain_gap.outputs.residual_discriminability` is quoted.

## All outputs

:::{astra} outputs
:::
