# Publication boundary

This repository distributes implementation source and generic configuration
examples. It does not distribute recordings, participant records, study results,
clinical annotations, pretrained models, fitted normalization values, healthy
templates, marker-coordinate arrays, or acquisition files.

The original recordings are protected. Sharing requires prior approval from the
hospital committee, which has not been obtained. The dataset is not available
through this repository. Publishing source code does not authorize access to
the dataset.

## Local execution

Use newly assigned opaque identifiers for any locally supplied participants and
recordings. Keep the identity mapping, dates, treatment histories, grading
records, and original acquisition filenames outside the checkout. The program
does not need names or initials. Split membership and optional side labels are
supplied locally and are never part of this distribution.

Training and evaluation create derived measurements and fitted artifacts. Those
outputs can retain information about participants even when names have been
removed. Keep all inputs and outputs in ignored local directories or outside
the checkout. Do not upload them as workflow artifacts or attach them to issues.

The synthetic demonstration generates artificial arrays at execution time from
a random seed. It does not load, sample, perturb, or fit the protected dataset.
Synthetic files and fitted demonstration models are not committed.

## Release checks

Before committing, stage only reviewed source files and run:

```bash
python tools/audit_release.py --staged
```

The gate allows only named source/documentation locations, checks exact index
blobs, rejects data/binary paths, and detects common credentials, personal
filesystem paths, email addresses, and calendar dates. It intentionally does
not follow ignored output directories. Staging is an explicit boundary: an
unexpected file is rejected even if it was force-added.

Maintainers can additionally pass a private newline-separated identifier list
with `--private-identifiers`. That list and any `--report` destination must
remain outside the checkout. Findings identify file, line, and rule, without
printing matched confidential text. Public automation does not contain the
private identifier list.

Short identifiers can collide with ordinary words, units, or numerical variable
names. A separate private `--reviewed-findings` file can record individually
reviewed semantic collisions. Each exception is bound to an exact file, line,
and line hash. Editing that line requires review again. This mechanism applies
only to identifier collisions, and cannot waive data-file or credential checks.

An automated scan complements source review. It cannot establish anonymity of
arbitrary arrays, text, models, or future submissions. Review every addition
before publication, including commit author and committer identity.
