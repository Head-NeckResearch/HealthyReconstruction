# Public implementation and scientific scope

This release is a selective portable adaptation of the final article pipeline.
It includes source code and hyperparameter settings, with no study observations,
patient or healthy-participant metadata, trained parameters, facial template,
clinical history, or results. The synthetic demo generates mathematical curved
point lattices at runtime. It is an integration check, not a clinical validation.

## Preprocessing and reference

Dental triad stabilization removes framewise rigid head motion. A shared
90-degree X/Y/Z canonical rotation is applied to stabilized face coordinates.
One robust proper rigid registration aligns the quality-reviewed actual neutral
to an explicit healthy-development neutral template; it is held fixed over the
recording. Registration retains trimmed RANSAC (four-point hypotheses, 10%
trimming), the original optional Z-axis half-turn check, and three Huber
refinement iterations with the 1.5 mm constant. At least 60 complete neutral
correspondences are required by the default settings.

The selected raw neutral stays independent of cropping. If no reviewed crop is
supplied, mean supported displacement energy determines the crop using the
5th-percentile threshold, up to five inactive rows bridged, the component
containing maximum energy, and expansion to at least 70 rows when available.
All targets are resampled to 100 equally spaced phase positions. An exact
observed point, or interpolation between two adjacent completely observed raw
rows, is supported. Missing-gap interpolation can be tracked separately but is
excluded from core neural inputs, targets and scoring. Neutral validity is
intersected with trajectory validity. There is no forced-zero initial motion or
endpoint extrapolation.

Healthy-development neutral coordinates and neutral-relative motions are
aggregated separately with coordinatewise 10%–90% finite quantile trimming.
The reference includes optimizer and early-stopping validation participants;
external evaluation participants never contribute. The validation partition
therefore measures early stopping and is not a fully nested reference estimate.

## Architecture and fitting

The static network concatenates 105 XYZ coordinates and their validity bits,
then uses fully connected widths `420→256→256→48→256→256→315`, GELU activations,
and encoder dropout 0.2. Corruption independently masks markers (0.30), optionally
one index half (0.20), and adds normalized Gaussian noise (0.03). The half mask
is an index partition inherited from the implementation, not a supplied
anatomical side annotation.

Each task has a separate temporal network. Its input has 420 channels; temporal
convolutions use hidden width 160, bottleneck width 40, kernel 5 and two
stride-two downsampling layers. Linear interpolation restores first the
half-length and then the full 100-frame length before decoding to 315 channels.
Encoder layers include batch normalization, GELU and dropout 0.2. Denoising
masks points (0.10), whole marker trajectories (0.35), and up to two temporal
blocks (probability 0.50 each, length 8–25), with normalized Gaussian noise 0.02.
The objective is coordinate-weighted strict-mask MSE. Velocity and acceleration
loss coefficients are zero.

Both branches use AdamW, learning rate 0.001, weight decay 0.0001, up to 300 epochs,
patience 40 and minimum validation improvement `1e-6`. Batch sizes are 16 static
and 8 temporal. The optimizer-only coordinate mean and standard deviation are
computed before any normalization; missing inputs become normalized zero.
Uncorrupted participant-exclusive validation selects a cloned best epoch.
The static RNG seed is 13 and the temporal RNG seed is reset to 17 per task.
Batch losses are aggregated by squared-error numerator and supported coordinate
count. Gradient norm clipping uses 5. Deterministic PyTorch algorithms are enabled.

Portable settings use CPU for both branches. The archived study static branch
used CUDA and temporal training used CPU, because deterministic CUDA linear
interpolation backward was unavailable in that software environment. An explicit
configuration change is needed for CUDA. Matching seeds does not guarantee
bitwise training identity across hardware, PyTorch versions, input order or
backends. No silent device fallback is performed.

## Final reconstruction, comparators and descriptors

The primary reference is exactly the sum of the decoded static neutral and
motion, `N_AE + X_AE`. Both identity and mean-motion blending coefficients are
zero. Earlier production experiments used blends, but those are not the final
article estimator and are not enabled by this release.

The comparators preserve the final article equations: observed neutral plus the
same decoded motion (without static projection), decoded neutral plus observed
motion (without temporal projection), healthy mean neutral plus healthy mean
motion, and target neutral plus healthy mean motion. The temporal ablation is a
static-residual comparator; it is not an independently learned healthy motion.

Scores use identical whole-XYZ support across all methods, with at least two
supported phase times per marker. Each coordinate trajectory is interpolated
only to stabilize filtering, processed with the original fourth-order
Butterworth/filter routine, and unsupported observations restored to NaN. The
`fs=100`, cutoff `8` convention corresponds to 0.08 cycles per normalized phase
sample; it is not a physical 8 Hz filter on variable-duration raw recordings.
The whole-face score averages each marker's supported-time Euclidean absolute
trajectory error, then averages markers.

The geometric map code preserves the study equations: absolute trajectory
error, reduced displacement amplitude, excess displacement amplitude, and
positive counterdirection component. Each trajectory uses its own neutral
origin. Direction is available only where observed and reference displacement
norms meet the configured floor, 1 mm by default, with 0.5 and 2 mm sensitivity
settings. Missing direction support is not interpreted as normal movement. No
marker side map or affected-side label is embedded; authorized callers can
provide generic marker-index groups to the independent regional reducers.

## Adaptation provenance and limits

Algorithm routines were selected from `train_static_neutral_ae_v3.py`,
`train_dynamic_motion_projector_v3.py`,
`align_and_mean_movement_reference_split_1b.py`,
`neutral_anchored_preprocessing.py`, `build_reviewed_neutral_dataset_v2.py`,
`build_reviewed_production_dataset.py`, `train_reviewed_models.py`,
`eval_healthy_reference_metrics.py`, `tbme_blending_kinematic_maps.py`, and the
final `tbme_ae_article_revision.py`. No source module with embedded private
configuration or data is copied wholesale into the release.

Public orchestration uses minimal NPZ input and opaque-token manifests instead
of dataset-specific CSV discovery. Its default RANSAC seed derives from a
canonical numeric-array hash, not the historical CSV-byte hash. An optional
private `alignment_seed` override and explicit template token preserve those
choices when available. Private study split assignments, neutral review choices,
crops, reference identity and training input order are needed to reproduce that
particular run. Omitting them creates another valid execution of the method,
not a claim of byte-for-byte reproduction of the article. The repository is
sufficient for deterministic synthetic smoke checks and for rerunning the
method on privately held compatible recordings. No historical performance
claim is inferred from the smoke test.
