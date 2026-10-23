# HealthyReconstruction

Code for healthy-reference reconstruction and quantitative analysis of 3D facial
movement. A masked static autoencoder estimates resting geometry, and
task-specific temporal autoencoders estimate displacement trajectories. The
reference is their direct sum. Statistical references and branch ablations use
the same evaluation support.

The analysis module includes regional excursion/direction descriptors,
participant-cluster bootstrap comparisons, aggregation sensitivity, participant
score influence, and descriptions across repeated visits.

**This is a code-only release. No recordings, participant metadata, clinical
records, derived study tables, fitted templates, normalization arrays, or trained
weights are distributed.** The public demonstration creates artificial inputs
from mathematical functions and a random seed at execution time.

## Install

The release was verified with Python 3.13.3 on Linux using CPU execution. Create
a virtual environment, activate it, then install:

```bash
python -m pip install -r requirements-reproducible.txt
python -m pip install --no-deps -e .
```

The requirements file pins the verified direct dependencies. An existing
compatible environment can instead install the package with
`python -m pip install -e .`. Other Python/library/hardware combinations have
not been verified by this release.

## Run without participant data

Choose a fresh local output directory:

```bash
healthy-reconstruction demo --output private/synthetic-demo --epochs 2 --tasks 5
healthy-reconstruction analyze --scores private/synthetic-demo/results/scores.json --output private/synthetic-demo/analysis.json --iterations 200 --seed 0
python -m unittest discover -s tests -v
```

The demonstration prepares generated point lattices, trains both branches,
predicts all comparison methods, and writes a smoke-test summary. It uses small
networks, short training, and fewer registration trials to exercise the workflow
quickly. It is an implementation check and provides no evidence of clinical
performance. Its inputs and outputs remain local and ignored by Git.

## Run with authorized private inputs

Convert authorized recordings locally to the numerical input format described
in [input_format.md](docs/input_format.md). Supply new opaque participant/sample
tokens, explicit optimization/validation/evaluation assignments, selected resting
frames, and any private acquisition decisions. The manifest schema rejects
additional fields such as names, initials, calendar dates, or clinical history.

```bash
healthy-reconstruction prepare --manifest private/manifest.json --output private/prepared --config config/paper.example.json
healthy-reconstruction train --dataset private/prepared --output private/models --config config/paper.example.json
healthy-reconstruction predict --dataset private/prepared --models private/models --output private/results --config config/paper.example.json --device cpu
healthy-reconstruction analyze --scores private/results/scores.json --output private/analysis.json --iterations 5000 --seed 0
```

Use fresh output directories; existing preparation, training, and prediction
directories are protected against accidental overwrite. The paper configuration
contains methodological settings and no participant assignments. Reconstruction
uses raw decoded branches without input blending or temporal-loss regularization.
GPU execution can be selected in a private configuration and at prediction time.

## Reproducibility and interpretation

- [Methods](docs/methods.md): preprocessing, validity masks, model fitting, and
  inference conventions.
- [Input format](docs/input_format.md): local numerical arrays, ordinal visits,
  opaque manifests, template selection, and alignment seed overrides.
- [Analysis](docs/analysis.md): score aggregation, shared bootstrap draws,
  regional contrasts, and follow-up calculations.
- [Provenance](docs/provenance.md): original algorithm sources, portable
  adaptations, dependency versions, and limits on exact reproduction.
- [Privacy](docs/privacy.md): the publication boundary and release audit gate.

Exact manuscript values require the protected dataset, original manual decisions,
private splits, and reference inputs. They cannot be reproduced from the public
repository alone. The synthetic route and analytical tests reproduce software
behavior without those inputs. Scores describe geometric departures from an
estimated reference; they are not clinical grades or validated treatment effects.

## Data availability

The original dataset is protected under French law. Sharing requires prior
approval from the hospital committee, which has not been obtained. The dataset
cannot currently be shared and is not provided by this repository.

## Release checks

Stage reviewed source changes, then run the fail-closed file/privacy audit:

```bash
python tools/audit_release.py --staged
```

The initial publication was additionally reviewed using a private identifier
dictionary kept outside the repository. The test suite runs source checks
and generated-input tests without access to the dataset. Automated scans
supplement manual review; keep all locally generated results private.

## License

The repository owner has not yet specified a software license. Contact the
owner through GitHub regarding reuse. No dataset license is granted.
