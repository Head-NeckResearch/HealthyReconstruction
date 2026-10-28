# HealthyReconstruction

Healthy-trained autoencoder reconstruction for descriptive quantification of
three-dimensional facial movement. A static AE reconstructs neutral morphology,
and task-specific temporal AEs reconstruct displacement sequences. Their sum is
the learned reference used to describe geometric deviations in millimeters.

The workflow addresses three questions:

1. **Cohort separation:** compare AE residuals with statistical references and
   branch ablations, using shared participant-bootstrap draws and alternative
   participant/visit aggregation policies.
2. **Regional movement:** describe excess, reduced and counter-direction
   displacement relative to the AE reconstruction, with coverage and paired
   side contrasts.
3. **Follow-up:** summarize AE residuals globally and by task across visits,
   together with available within-visit take variation.

Counter-direction uses the original **fixed 1-mm amplitude eligibility rule**
for both observed and reconstructed displacement. Pattern counts use a fixed
`1e-6`-mm numerical tie tolerance. The release contains no alternative-floor
sweep or additional side-contrast margin analysis.

## What is included

Source code, a paper-method configuration, analytical tests, and a generated-input
smoke demonstration. **No recordings, participant metadata, clinical records,
derived study tables, fitted templates, normalization arrays or trained weights
are distributed.** All generated outputs remain local and ignored by Git.

The reference is a model estimate learned from healthy observations. The software
provides descriptive geometric measurements; it is not a validated clinical
severity scale or a measure of treatment effectiveness.

## Install

The reference environment is Python 3.13.3 on Linux. In an activated virtual or
Conda environment, install the verified direct dependency versions and package:

```bash
python -m pip install -r requirements-reproducible.txt
python -m pip install --no-deps -e .
```

Alternatively, `python -m pip install -e .` uses the supported dependency ranges.
Other library and hardware combinations may produce different results.

## Try the workflow without participant data

Use a fresh output directory:

```bash
healthy-reconstruction demo --output private/synthetic-demo --epochs 2 --tasks 5
healthy-reconstruction analyze --scores private/synthetic-demo/results/scores.json --output private/synthetic-demo/analysis.json --iterations 200 --seed 0
python -m unittest discover -s tests -v
```

The demo generates mathematical point lattices at execution time, trains small
networks briefly, and exercises preparation, reconstruction and analysis. It
checks implementation behavior and makes no claim about clinical performance.

## Run with authorized private recordings

Prepare local numerical inputs as described in [the input format](docs/input_format.md).
Supply opaque participant/sample tokens, explicit development/evaluation splits,
reviewed resting frames and acquisition decisions. The manifest rejects extra
fields such as names, calendar dates and clinical histories.

```bash
healthy-reconstruction prepare --manifest private/manifest.json --output private/prepared --config config/paper.example.json
healthy-reconstruction train --dataset private/prepared --output private/models --config config/paper.example.json
healthy-reconstruction predict --dataset private/prepared --models private/models --output private/results --config config/paper.example.json --device cpu
healthy-reconstruction analyze --scores private/results/scores.json --output private/analysis.json --iterations 5000 --seed 0
```

Preparation, training and prediction protect existing output directories against
overwrite. The configuration preserves the original architecture, denoising,
early stopping and direct decoded reconstruction, without inference-time input
blending or temporal derivative-loss regularization.

All five residual methods are retained for the cohort-separation experiment.
Regional map outputs and the file-based follow-up/within-visit reports use AE
only. `analyze` requires an `ae` score column for those reports; generic
benchmark-only score vectors can be compared using the `cluster_auc` helper. Counter-direction eligibility remains
fixed at 1 mm, and nonmatching or obsolete multi-floor configurations are rejected.

## Methods and reproducibility

- [Methods](docs/methods.md): preprocessing, support masks, architecture,
  training, reconstruction and fixed-rule descriptors.
- [Input format](docs/input_format.md): numerical arrays, opaque manifests,
  resting-frame selection and explicit local assignments.
- [Analysis](docs/analysis.md): participant-level AUC uncertainty, regional
  patterns and AE-only follow-up calculations.
- [Provenance](docs/provenance.md): source adaptations and reproducibility limits.
- [Privacy](docs/privacy.md): code-release boundary and audit procedure.

Exact article values require the protected recordings, original manual decisions,
private assignments and fitted inputs. They cannot be reproduced from this
code-only repository alone. Generated-input tests verify software behavior.

## Data availability

The original dataset is protected under French law and cannot currently be
shared without hospital committee approval, which has not been obtained. This
repository does not distribute that dataset or grant a dataset license.

## Release verification

After staging reviewed source changes, run:

```bash
python tools/audit_release.py --staged
python -m unittest discover -s tests -v
```

The audit checks the staged source files and rejects unexpected artifacts or
potentially sensitive content. Automated checks supplement manual review; keep
all private inputs and generated results outside the published file set.

## License

The repository owner has not specified a software license. Contact the owner
through GitHub regarding reuse. No license to the protected dataset is granted.
