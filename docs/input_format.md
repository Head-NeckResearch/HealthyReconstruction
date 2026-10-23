# Private input format

The repository contains no acquired coordinates, reference template, model
weights, participant records, clinical information, or study-specific split.
Generate the artificial example with `healthy-reconstruction demo --output
/tmp/reconstruction-demo`; its files illustrate the schema using mathematical
point lattices. Keep all real inputs and generated outputs in a private folder
outside the checkout. Input preparation is local; no command sends them online.

## Numerical recordings

Provide one NumPy `.npz` per take. It must contain exactly one key, `points`, a
real numerical array in millimetres. Object arrays and extra fields are rejected.

- `coordinate_system = "world"`: `(raw_frames, 108, 3)`, with the first three
  markers being the rigid dental triad and the remaining 105 the facial markers.
  Their order must be identical across every take. Each frame is transformed into
  its own dental coordinate system; missing/degenerate triads invalidate that
  whole frame.
- `coordinate_system = "dental"`: `(raw_frames, 105, 3)`, already stabilized in
  the dental frame, before the shared canonical 90-degree X/Y/Z rotation. Do not
  apply that rotation upstream a second time. This option does not estimate or
  validate dental stabilization for you.

Use NaN for missing complete XYZ points. Infinite coordinates are rejected.
Recordings must have consecutive, uniformly sampled raw rows at the configured
sampling frequency. Check acquisition discontinuities before conversion. The
public loader does not infer proprietary CSV headers, identities, dates, or
metadata from file names.

## Minimal private manifest

A JSON object has only these top-level fields:

| Field | Type and meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `coordinate_system` | `world` or `dental`, as above. |
| `records` | List of private per-take entries described below. |
| `template_sample_id` | Optional opaque token selecting an included healthy development recording as the registration template. Required to reproduce a particular historical template. |

Each entry accepts only these fields; unrecognized fields, including clinical
fields, are rejected:

| Field | Type and meaning |
| --- | --- |
| `sample_id` | Unique random lowercase 32-character hexadecimal token. |
| `participant_id` | Random lowercase 32-character hexadecimal token, reused across the same person's takes and visits. |
| `role` | `optimizer`, `validation`, `healthy_eval`, or `pathological_eval`. One participant must have exactly one role. |
| `task` | `M1`, `M2`, `M3`, `M4`, or `M5`. |
| `visit_index` | Positive, 1-based ordinal visit; no dates or clinical history. |
| `file` | Relative `.npz` path inside the private manifest directory. |
| `neutral_raw_index` | Quality-reviewed, zero-based actual raw resting-frame row. |
| `crop_start`, `crop_stop` | Optional integer half-open raw row bounds; supply both or neither. Without them the study energy-window heuristic is used. |
| `alignment_seed` | Optional integer in `[0, 2**32)` for registration. Can preserve the historical seed privately without publishing its source hash or identity. |

Generate random tokens independently of names, initials, dates, medical numbers,
and original filenames. A token-shaped identifier alone does not establish
anonymity: the private mapping must remain outside the repository. Even this
minimal runtime manifest and its role assignments are private inputs and must
not be committed.

Both development roles must be present for each selected task. Supply the
original private assignments when reproducing a historical split. The reusable
`grouped_participants` helper implements the seeded participant-level holdout
without embedding any original participant. It is optional; the CLI uses explicit
manifest roles to keep the intended partition reviewable.

Exact duplicate numerical recordings are excluded once in stable token order.
Identical content assigned to different participants, roles, or tasks is an
error. The template is selected only from healthy development. Without an
explicit token, the most complete development neutral is used, with a token
order tie break. Thus an unspecified template is portable, but does not promise
the template used in the original study.

## Prepared and generated files

Preparation creates local `samples/*.npz`, task references, a minimal runtime
manifest and configuration. Training adds checkpoints, training histories and
optimizer-fitted normalization. Prediction writes all five comparison scores and
geometric maps; analysis writes aggregate results. These files may still contain
sensitive facial information and must remain private. No real example of any of
them is distributed in this repository. The ignore rules reduce accidental
staging; they are not a substitute for reviewing files before publication.

A prepared sample contains `N` `(105,3)`, `X` `(100,105,3)`, strict `M`
`(100,105)`, `neutral_mask` `(105,)`, positions `P`, imputation flags, and phase
positions. Missing target coordinates have a false mask. Gap-imputed values do
not enter core inputs, loss or scores. `N` stays the actual selected raw neutral;
its index need not belong to the crop, and `X[0]` is generally nonzero.

No raw dataset, calibrated acquisition template, study participant split, marker
annotation table, or historical model weights are supplied. Consequently the
repository can reproduce its synthetic workflow and run the method on an
authorized private dataset, but cannot recreate the article's numerical cohort
results by itself.
