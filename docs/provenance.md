# Implementation provenance and reproducibility

This release selectively adapts the research implementation. Original working
directories and their history are not imported into the public repository.
Participant-specific defaults, mappings, case selections, clinical processing,
private paths, and cached measurements are excluded.

## Algorithm sources

The implementation was traced through these original source modules:

- `neutral_anchored_preprocessing.py`, `build_reviewed_production_dataset.py`,
  and `build_reviewed_neutral_dataset_v2.py`: neutral-anchored spatial and
  temporal preparation and validity support.
- `train_static_neutral_ae_v3.py` and
  `train_dynamic_motion_projector_v3.py`: network definitions and masking/noise
  corruption operations.
- `train_reviewed_models.py`: participant-disjoint optimization/early-stopping
  partitions, optimization-only normalization, deterministic training, and
  retained best-epoch parameters.
- `evaluate_reviewed_pipeline.py` and `tbme_ae_article_revision.py`: prediction
  assembly, evaluation support, residuals, and regional displacement maps.
- The follow-up, regional-pattern, and aggregation-sensitivity analysis code:
  participant-cluster uncertainty, score influence, side contrasts, and ordinal
  longitudinal summaries.

The article's final reference is the decoded neutral geometry plus decoded
movement displacement. The descriptive workflow uses the original fixed 1-mm
direction eligibility rule and numerical tie tolerance; regional maps and
file-based follow-up use AE only. Statistical references and ablations support
the cohort-separation experiment. Historical input-blending options in earlier scripts
are not part of this release's reference definition. Temporal derivative
regularization is disabled in the article configuration.

## What can be reproduced

The public synthetic route exercises preparation, model training, inference,
mask handling, and analysis without access to a real participant. Analytical
tests verify formulas and important invariants. They test implementation
behavior; they do not validate clinical performance.

Users with authorized local recordings can supply anonymous manifests, their
own marker layout, explicit participant splits, resting-frame selections, and
a private healthy-development template. These decisions and inputs materially
affect the fitted model and resulting scores. Default synthetic inputs or a
newly generated split do not reproduce the article's historical inputs.

Exact manuscript values require the protected recordings, original manual
decisions and private assignments. Their absence prevents independent numerical
reproduction of the original cohort results from this repository alone. No
trained weights, subject-level results, or fitted reference arrays are supplied.

The reference dependency versions are listed in
`requirements-reproducible.txt`. The file pins direct dependencies; it is not
a complete operating-system or transitive-dependency lock. Deterministic
settings improve repeatability within a fixed environment. Results can differ
across hardware, operating systems, library versions, and accelerator kernels.

## Redistribution status

No software license has yet been specified by the repository owner. This
release does not assign an open-source license or a license to the protected
dataset. Contact the repository owner through GitHub regarding reuse.
