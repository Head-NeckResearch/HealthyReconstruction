# Portable statistical analyses

`healthy_reconstruction.analysis` implements analyses of private, frozen scores
and geometric maps. It contains no cohort observations, patient assignments,
recording names, dates, clinical histories, or fitted model weights. Tests create
small mathematical examples in memory. Individual summaries produced by these
functions are private outputs and must remain outside the public repository.

The release preserves the calculations while removing study-specific loading,
selection, identifier mappings, and clinical linkage. It does not reproduce the
paper's numerical results without the unavailable private inputs. Exact seeded
reproduction also requires the original input participant order, scores, task
selection, software versions, and preprocessing/model configuration.

## Input scores and repeated takes

A `ScoreRecord(participant_id, group, visit_index, task, scores)` contains one
participant/task/visit score vector, following an explicitly supplied list of
method names. Identifiers must be local pseudonyms. `visit_index` is a positive
ordinal, not a date or elapsed time. Groups default to `reference` and
`comparison`; callers can choose different group names locally. Scores are
finite, higher-is-more-abnormal reconstruction residuals. Methods must be paired
on the same task units. Duplicate units and incomplete method vectors raise an
error instead of silently changing the analysis set.

Use `aggregate_take_scores(takes_by_methods)` to obtain the componentwise median
over repeated takes before constructing records. Repetitions do not become
independent participants. Input array units should match the preprocessing units;
the functions do not infer physical units or convert them.

`analyze_file(scores_path, output_path, iterations=5000, seed=0, tasks=None, ...)`
reads the private prediction command's JSON envelope (`schema_version`, `methods`,
`records`). It validates unique take identifiers, aggregates take medians, and
writes a private report containing the AUC analyses, score-omission diagnostic,
ordinal follow-up, and repetition variation. It does not forward extra input
fields. Set both group names to the values used by the private input. Methods
follow the declared header; task order defaults to first appearance unless
explicitly supplied. Regional patterns require maps and local region assignments
and cannot be recovered from whole-face scores alone. The file loader explicitly
marks follow-up unavailable if comparison visits have incomplete task sets.

```python
from healthy_reconstruction.analysis import ScoreRecord, cluster_auc

# Construct records from approved, private local scores; nothing is loaded from
# the network or from a bundled participant table.
records = []  # populate locally
# result = cluster_auc(records, methods=("model", "baseline"),
#                      tasks=("task_one", "task_two"), seed=0)
```

## RQ1: discrimination and aggregation robustness

`cluster_auc` implements four estimands, in `POLICIES` order:

1. `original_session_units`: all participant/task/visit units, so participants
   with more visits contribute more score pairs.
2. `participant_median`: one median across visits per participant/task/method.
3. `first_comparison_visit_reference_median`: first ordinal visit of each
   comparison participant and participant medians in the reference group. If a
   task is absent at the first visit, it remains absent; a later visit is not
   substituted.
4. `equal_subject_weighted_sessions`: mean pairwise concordance across visits
   within each reference/comparison participant pair, then equal weighting of
   supported participant pairs.

For each task/method, AUC is the probability that a comparison score exceeds a
reference score, with one half for an exact tie. The final `macro_mean` is the
arithmetic mean of task-specific AUCs; it is not an AUC after pooling tasks.
Missing task observations are not imputed.

The bootstrap resamples participants independently within each group, retaining
all their methods/tasks/visits together. The same sampled participant counts are
used for every method, task, and estimand. A draw lacking either group's support
for any task/estimand is rejected jointly. This generalization also handles a
missing first-visit task; where the first-visit task set is complete, it preserves
the original calculation. Percentile intervals use NumPy's default linear
quantiles. `AUCResult.paired_difference(a, b)` calculates paired differences from
those shared draws. A 95% interval excluding zero would describe score-level
uncertainty conditional on the trained models and available cohort; it does not
include training-set/model-fitting uncertainty.

Participant order is first appearance in the supplied records. This order affects
the exact seeded random draws, though it does not affect the point estimate.
Supply explicit task/method order and seed for reproducibility. The default
number of draws is a computation setting, not a statement about cohort size.

`leave_one_subject_out` deletes each participant's frozen scores in turn. It does
**not** retrain a model or perform model-validation cross-validation. Unsupported
task estimates and their macro-average become NaN. Returned individual labels and
estimates are private and should not be committed.

## RQ2: regional patterns, eligibility, and margins

First compute maps with the inference module using filtered observed/reference
coordinates and each sequence's own stored neutral. `regional_summary(maps,
marker_indices)` computes the finite mean across frames separately for each
marker, followed by a finite mean across selected markers. This is equal-marker
weighting; flattening all valid frame/marker cells would change the estimand when
markers have unequal missingness. `aggregate_regional_takes` then uses medians of
per-take regional values and coverages separately.

Coverage is the number of finite map entries divided by the entire selected
frame-by-marker grid. Missing coordinates and ineligible directional frames both
reduce it. The inference map function excludes counter-direction calculations
when either observed or reference displacement amplitude falls below the chosen
direction floor. Unavailable values remain NaN; they never establish normal
function. The amplitude maps themselves do not depend on this floor. Repeat map
calculation locally at each prespecified floor to assess that sensitivity.

`side_contrasts(a, b)` takes a locally assigned generic side A and side B. Its
default signs are A minus B for `reduced_movement` and `counter_direction`, and B
minus A for `excess_movement`. No clinical side assignment is bundled. Pair the
two side values within each visit first, then use `summarize_contrasts` to obtain
the finite median of paired contrasts across visits. This generally differs from
subtracting side medians after aggregation. The alternative earliest-visit
summary uses the smallest recorded visit index, including a missing descriptor
if it was missing at that visit.

`pattern_frequencies(participants_by_metrics, margin=...)` returns the count
strictly above the chosen margin among finite eligible contrasts, separately for
each metric and jointly for all metrics. Numerical ties do not support the
ordering. Joint eligibility requires every requested descriptor to be finite.
Wilson intervals describe participant frequencies; they are not estimates of
diagnostic sensitivity, population prevalence, or treatment efficacy. A contrast
margin and a directional amplitude floor are different quantities. Margins are
descriptive unless independently validated as clinical/error thresholds. State
all tested margins/floors rather than selecting the best one.

## RQ3: ordinal follow-up and within-visit variation

`summarize_followup` reports each visit's arithmetic mean over the declared task
scores and first-to-last task/global differences. Every included visit must
contain every declared task. Visits are ordered by ordinal index only, and no
elapsed-time, treatment, or clinical-grade fields are consumed or released.
Single-visit participants have visit summaries but no first-to-last endpoint.
Percentage change is unavailable when the baseline equals zero. A smaller
residual denotes greater similarity to the selected healthy reference, not
validated clinical recovery.

`within_visit_variability` consumes private `(participant_id, visit_index, task)`
keys and `(takes, methods)` arrays. It reports take medians and interquartile
ranges using linear quantiles; single-take IQRs are unavailable. Within-visit
take dispersion does not estimate between-visit repeatability, a minimal
detectable change, or uncertainty of a longitudinal change.

## Validation and limitations

Run the package's standard-library test suite. Analysis tests independently
enumerate score pairs for each aggregation policy, check shared participant
bootstrap draws and task-support rejection, and verify ties, missing values,
fixed-task means, nested regional weighting, contrast ordering, and ordinal
endpoints. These tests validate mathematical behavior on generated inputs; they
do not provide external replication or clinical validation.
