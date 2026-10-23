"""Portable frozen-score, regional, and ordinal follow-up analyses.

No observations, cohort membership, side assignments, dates, or model weights
are included. Runtime records and returned individual summaries remain private.
Higher scores are assumed to indicate greater departure from the reference.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np


POLICIES = (
    "original_session_units",
    "participant_median",
    "first_comparison_visit_reference_median",
    "equal_subject_weighted_sessions",
)
PATTERN_METRICS = ("reduced_movement", "counter_direction", "excess_movement")


@dataclass(frozen=True)
class ScoreRecord:
    """One paired score vector for one participant, ordinal visit, and task.

    ``scores`` follows the caller's ``methods`` order. Aggregate repeated takes
    first with ``aggregate_take_scores``. Group names are caller-defined.
    """

    participant_id: str
    group: str
    visit_index: int
    task: str
    scores: tuple[float, ...]


@dataclass
class AUCResult:
    """Arrays have policy, task (including final macro-average), method axes."""

    policies: tuple[str, ...]
    tasks: tuple[str, ...]
    methods: tuple[str, ...]
    estimate: np.ndarray
    bootstrap: np.ndarray
    reference_draw_counts: np.ndarray
    comparison_draw_counts: np.ndarray
    confidence: float

    def interval(self) -> np.ndarray:
        """Percentile bounds, shape (2, policy, task, method)."""
        if self.bootstrap.shape[0] == 0:
            raise ValueError("No bootstrap draws were requested")
        alpha = (1.0 - self.confidence) / 2.0
        return np.quantile(self.bootstrap, [alpha, 1.0 - alpha], axis=0)

    def paired_difference(self, method_a: str, method_b: str) -> dict[str, np.ndarray]:
        """Paired AUC differences use exactly the same cluster draws."""
        a, b = self.methods.index(method_a), self.methods.index(method_b)
        result = {"estimate": self.estimate[..., a] - self.estimate[..., b]}
        if self.bootstrap.shape[0]:
            alpha = (1.0 - self.confidence) / 2.0
            difference = self.bootstrap[..., a] - self.bootstrap[..., b]
            result["interval"] = np.quantile(difference, [alpha, 1.0 - alpha], axis=0)
        return result


def _finite_mean(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    count = finite.sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _finite_median(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def aggregate_take_scores(take_scores: np.ndarray) -> np.ndarray:
    """Median across repeated takes, preserving method pairing.

    Input has shape (takes, methods). Every score must be finite; partially
    observed method vectors must be resolved by the caller before comparison.
    """
    scores = np.asarray(take_scores, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("take_scores must be a nonempty finite (takes, methods) array")
    return np.median(scores, axis=0)


def _prepare(records, methods, tasks, reference_group, comparison_group):
    records, methods, tasks = list(records), tuple(methods), tuple(tasks)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    if not tasks or len(set(tasks)) != len(tasks) or "macro_mean" in tasks:
        raise ValueError("tasks must be nonempty, unique, and exclude macro_mean")
    if reference_group == comparison_group:
        raise ValueError("Reference and comparison groups must differ")
    groups = (reference_group, comparison_group)
    ids = {group: [] for group in groups}
    membership, seen = {}, set()
    by_unit = {}
    for row in records:
        if row.group not in groups or row.task not in tasks:
            raise ValueError("Record group/task is absent from the declared analysis")
        if not isinstance(row.participant_id, str) or not row.participant_id:
            raise ValueError("participant_id must be a nonempty local pseudonym")
        if isinstance(row.visit_index, bool) or not isinstance(row.visit_index, (int, np.integer)) or row.visit_index < 1:
            raise ValueError("visit_index must be a positive integer")
        values = np.asarray(row.scores, dtype=float)
        if values.shape != (len(methods),) or not np.isfinite(values).all():
            raise ValueError("Every score vector must have one finite value per method")
        old = membership.setdefault(row.participant_id, row.group)
        if old != row.group:
            raise ValueError("A participant cannot occur in both groups")
        if row.participant_id not in ids[row.group]:
            ids[row.group].append(row.participant_id)
        key = (row.group, row.participant_id, row.visit_index, row.task)
        if key in seen:
            raise ValueError("Duplicate visit/task unit; aggregate repeated takes first")
        seen.add(key)
        by_unit[key] = values
    if any(not ids[g] for g in groups):
        raise ValueError("Both groups need at least one participant")
    first = {(g, p): min(k[2] for k in by_unit if k[:2] == (g, p)) for g in groups for p in ids[g]}
    grouped = {}
    for g in groups:
        for p in ids[g]:
            for task in tasks:
                keys = sorted(k for k in by_unit if k[:2] == (g, p) and k[3] == task)
                grouped[(g, p, task)] = np.asarray([by_unit[k] for k in keys]).reshape(-1, len(methods))
    nh, nc = (len(ids[g]) for g in groups)
    kernels, supports_h, supports_c = [], [], []
    for policy in POLICIES:
        values = {}
        for (g, p, task), a in grouped.items():
            if len(a) and (policy == "participant_median" or (policy == POLICIES[2] and g == reference_group)):
                a = np.median(a, axis=0, keepdims=True)
            elif policy == POLICIES[2] and g == comparison_group:
                value = by_unit.get((g, p, first[(g, p)], task))
                a = value[None, :] if value is not None else np.empty((0, len(methods)))
            values[(g, p, task)] = a
        kernel = np.zeros((len(tasks), nh, nc, len(methods)))
        hs, cs = np.zeros((len(tasks), nh)), np.zeros((len(tasks), nc))
        equal = policy == "equal_subject_weighted_sessions"
        for ti, task in enumerate(tasks):
            for hi, h in enumerate(ids[reference_group]):
                a = values[(reference_group, h, task)]
                hs[ti, hi] = bool(len(a)) if equal else len(a)
                for ci, c in enumerate(ids[comparison_group]):
                    b = values[(comparison_group, c, task)]
                    cs[ti, ci] = bool(len(b)) if equal else len(b)
                    if len(a) and len(b):
                        difference = b[None, :, :] - a[:, None, :]
                        concordance = (difference > 0).astype(float) + 0.5 * (difference == 0)
                        kernel[ti, hi, ci] = concordance.mean(axis=(0, 1)) if equal else concordance.sum(axis=(0, 1))
        kernels.append(kernel)
        supports_h.append(hs)
        supports_c.append(cs)
    return methods, tasks, ids, kernels, supports_h, supports_c


def _estimate(hcounts, ccounts, kernel, hs, cs):
    numerator = np.einsum("bi,tijm,bj->btm", hcounts, kernel, ccounts, optimize=True)
    denominator = (hcounts @ hs.T) * (ccounts @ cs.T)
    auc = np.divide(numerator, denominator[..., None], out=np.full_like(numerator, np.nan), where=denominator[..., None] > 0)
    # A missing task makes the macro-average unavailable: do not change weights.
    return np.concatenate([auc, auc.mean(axis=1, keepdims=True)], axis=1)


def cluster_auc(
    records: Iterable[ScoreRecord], methods: Sequence[str], tasks: Sequence[str], *,
    reference_group: str = "reference", comparison_group: str = "comparison",
    n_bootstrap: int = 5000, seed: int = 0, confidence: float = 0.95,
) -> AUCResult:
    """Four AUC estimands and shared participant-cluster bootstrap draws.

    Sampling is stratified by group, with replacement and original group sizes.
    A single draw carries all visits/tasks/methods of each sampled participant.
    Draws with no support for any declared task/policy are rejected jointly.
    Input participant first-appearance order is part of exact seeded reproduction.
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    prepared = _prepare(records, methods, tasks, reference_group, comparison_group)
    methods, tasks, ids, kernels, hsupports, csupports = prepared
    nh, nc = len(ids[reference_group]), len(ids[comparison_group])
    points = np.stack([_estimate(np.ones((1, nh)), np.ones((1, nc)), k, hs, cs)[0] for k, hs, cs in zip(kernels, hsupports, csupports)])
    if not np.isfinite(points).all():
        raise ValueError("Every task must be supported in both groups under all four policies")
    rng = np.random.default_rng(seed)
    hdraws, cdraws = [], []
    attempts = 0
    while len(hdraws) < n_bootstrap:
        attempts += 1
        if attempts > max(10000, 1000 * n_bootstrap):
            raise ValueError("Insufficient common task support for bootstrap resampling")
        hd = rng.integers(0, nh, nh)
        cd = rng.integers(0, nc, nc)
        if any(np.any(hs[:, hd].sum(axis=1) == 0) or np.any(cs[:, cd].sum(axis=1) == 0) for hs, cs in zip(hsupports, csupports)):
            continue
        hdraws.append(np.bincount(hd, minlength=nh))
        cdraws.append(np.bincount(cd, minlength=nc))
    hcounts = np.asarray(hdraws, dtype=int).reshape(n_bootstrap, nh)
    ccounts = np.asarray(cdraws, dtype=int).reshape(n_bootstrap, nc)
    if n_bootstrap:
        bootstrap = np.stack([_estimate(hcounts, ccounts, k, hs, cs) for k, hs, cs in zip(kernels, hsupports, csupports)], axis=1)
    else:
        bootstrap = np.empty((0, len(POLICIES), len(tasks) + 1, len(methods)))
    return AUCResult(POLICIES, tasks + ("macro_mean",), methods, points, bootstrap, hcounts, ccounts, confidence)


def leave_one_subject_out(
    records: Iterable[ScoreRecord], methods: Sequence[str], tasks: Sequence[str], *,
    reference_group: str = "reference", comparison_group: str = "comparison",
) -> list[dict]:
    """Omit each participant's scores; never refit or retrain a model.

    Each returned estimate has policy/task/method axes including macro_mean.
    Unsupported task estimates and their macro-average remain NaN.
    """
    methods, tasks, ids, kernels, hsupports, csupports = _prepare(records, methods, tasks, reference_group, comparison_group)
    nh, nc = len(ids[reference_group]), len(ids[comparison_group])
    full = np.stack([_estimate(np.ones((1, nh)), np.ones((1, nc)), k, hs, cs)[0] for k, hs, cs in zip(kernels, hsupports, csupports)])
    output = []
    for group in (reference_group, comparison_group):
        for index, participant in enumerate(ids[group]):
            hc, cc = np.ones((1, nh)), np.ones((1, nc))
            (hc if group == reference_group else cc)[0, index] = 0
            estimate = np.stack([_estimate(hc, cc, k, hs, cs)[0] for k, hs, cs in zip(kernels, hsupports, csupports)])
            output.append({"participant_id": participant, "group": group, "estimate": estimate, "change_from_full": estimate - full})
    return output


def summarize_followup(records: Iterable[ScoreRecord], methods: Sequence[str], tasks: Sequence[str]) -> dict[str, list[dict]]:
    """Task-balanced visit scores and ordinal first/last changes.

    No elapsed time or treatment variable is used. All requested tasks must be
    present at each included visit; missing tasks raise rather than alter weights.
    Returned individual records must not be published without authorization.
    """
    methods, tasks = tuple(methods), tuple(tasks)
    if not methods or len(set(methods)) != len(methods) or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("methods and tasks must be nonempty and unique")
    units, seen, membership = defaultdict(dict), set(), {}
    for row in records:
        if row.task not in tasks or row.visit_index < 1 or isinstance(row.visit_index, bool) or not isinstance(row.visit_index, (int, np.integer)):
            raise ValueError("Unknown task or invalid visit index")
        if not row.participant_id or membership.setdefault(row.participant_id, row.group) != row.group:
            raise ValueError("Invalid participant identifier or inconsistent group")
        value = np.asarray(row.scores, float)
        if value.shape != (len(methods),) or not np.isfinite(value).all():
            raise ValueError("Invalid paired method score vector")
        key = row.participant_id, row.visit_index, row.task
        if key in seen:
            raise ValueError("Duplicate visit/task score")
        seen.add(key)
        units[(row.participant_id, row.visit_index)][row.task] = value
    visits, endpoints = [], []
    by_participant = defaultdict(list)
    for (participant, visit), scores in sorted(units.items()):
        if set(scores) != set(tasks):
            raise ValueError("Every follow-up visit requires every declared task")
        matrix = np.stack([scores[t] for t in tasks])
        row = {"participant_id": participant, "visit_index": visit, "task_scores": matrix, "global_score": matrix.mean(axis=0)}
        visits.append(row)
        by_participant[participant].append(row)
    for participant, participant_visits in by_participant.items():
        if len(participant_visits) < 2:
            continue
        first, last = participant_visits[0], participant_visits[-1]
        delta = last["task_scores"] - first["task_scores"]
        baseline = first["global_score"]
        global_delta = last["global_score"] - baseline
        percent = np.divide(100 * global_delta, baseline, out=np.full_like(global_delta, np.nan), where=baseline != 0)
        endpoints.append({"participant_id": participant, "n_visits": len(participant_visits), "first_visit_index": first["visit_index"], "last_visit_index": last["visit_index"], "baseline_global_score": baseline, "last_global_score": last["global_score"], "global_delta": global_delta, "global_delta_percent": percent, "task_delta": delta, "n_tasks_larger": (delta > 0).sum(axis=0), "n_tasks_smaller": (delta < 0).sum(axis=0)})
    return {"visits": visits, "endpoints": endpoints}


def within_visit_variability(take_scores: Mapping[tuple[str, int, str], np.ndarray]) -> list[dict]:
    """Take count, median, and IQR per local participant/visit/task unit.

    Values are (takes, methods). Single-take units have an unavailable IQR,
    preventing a single observation from implying perfect repeatability.
    """
    output = []
    for (participant, visit, task), values in take_scores.items():
        values = np.asarray(values, float)
        if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
            raise ValueError("Take scores must be finite nonempty (takes, methods) arrays")
        quartiles = np.quantile(values, [0.25, 0.75], axis=0)
        output.append({"participant_id": participant, "visit_index": visit, "task": task, "n_takes": len(values), "median": np.median(values, axis=0), "iqr": quartiles[1] - quartiles[0] if len(values) > 1 else np.full(values.shape[1], np.nan)})
    return output


def regional_summary(maps: Mapping[str, np.ndarray], marker_indices: Sequence[int]) -> dict[str, dict[str, float]]:
    """Time mean per marker, then equal-marker mean within a generic region.

    ``maps`` contains (frames, markers) arrays from the inference map function.
    Coverage is finite entries / ALL selected frame-marker cells, retaining
    missing coordinates and direction-floor exclusions in the denominator.
    """
    indices = np.asarray(marker_indices)
    if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu" or len(set(indices.tolist())) != len(indices):
        raise ValueError("marker_indices must be nonempty unique integer indices")
    result = {}
    for metric, values in maps.items():
        values = np.asarray(values, float)
        if values.ndim != 2 or min(values.shape) < 1 or np.any(indices < 0) or np.any(indices >= values.shape[1]):
            raise ValueError("Maps must be nonempty (frames, markers) arrays with valid region indices")
        selected = values[:, indices]
        result[metric] = {"value": float(_finite_mean(_finite_mean(selected, axis=0))), "coverage": float(np.isfinite(selected).mean())}
    return result


def aggregate_regional_takes(takes: Sequence[Mapping[str, Mapping[str, float]]]) -> dict[str, dict[str, float]]:
    """Finite median of per-take regional values and coverages, separately."""
    if not takes or not takes[0] or any(set(t) != set(takes[0]) for t in takes):
        raise ValueError("All takes must contain the same nonempty metric set")
    return {metric: {key: _finite_median([take[metric][key] for take in takes]) for key in ("value", "coverage")} for metric in takes[0]}


def side_contrasts(side_a: Mapping[str, float], side_b: Mapping[str, float], *, metrics: Sequence[str] = PATTERN_METRICS, reverse: Sequence[str] = ("excess_movement",)) -> dict[str, float]:
    """Paired within-visit contrasts; generic side assignment is local input.

    Default sign is A minus B for reduced/counter-direction and B minus A for
    excess. This specifies a geometric ordering, not a diagnosis or severity.
    """
    result = {}
    for metric in metrics:
        a, b = float(side_a[metric]), float(side_b[metric])
        result[metric] = (b - a if metric in reverse else a - b) if np.isfinite([a, b]).all() else float("nan")
    return result


def summarize_contrasts(visits: Sequence[tuple[int, Mapping[str, float]]], metrics: Sequence[str] = PATTERN_METRICS) -> dict[str, dict[str, float]]:
    """Paired contrasts first, finite median across visits second.

    Earliest means smallest recorded ordinal visit, including missing values;
    it never selects the earliest nonmissing value separately for each metric.
    """
    if not visits or len({v for v, _ in visits}) != len(visits) or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v, _ in visits):
        raise ValueError("Provide unique positive ordinal visit indices")
    ordered = sorted(visits, key=lambda x: x[0])
    return {"participant_median": {m: _finite_median([r[m] for _, r in ordered]) for m in metrics}, "earliest_visit": {m: float(ordered[0][1][m]) for m in metrics}}


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson interval for descriptive participant frequencies."""
    if not 0 < confidence < 1 or not 0 <= k <= n or not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("Require integer 0 <= k <= n and 0 < confidence < 1")
    if n == 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(center - half), float(center + half)


def pattern_frequencies(contrasts: np.ndarray, *, metrics: Sequence[str] = PATTERN_METRICS, margin: float = 1e-6, confidence: float = 0.95) -> dict[str, dict]:
    """Per-metric and joint support among eligible participant contrasts.

    Rows are participants and columns follow metrics. Strictly greater than
    ``margin`` supports the ordering; missing descriptors are never zeros.
    A margin is descriptive and has no implied clinical or error threshold.
    """
    values, metrics = np.asarray(contrasts, float), tuple(metrics)
    if values.ndim != 2 or values.shape[1] != len(metrics) or not metrics or len(set(metrics)) != len(metrics) or "all_metrics" in metrics:
        raise ValueError("Expected (participants, unique metrics) array")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    output = {}
    for index, metric in enumerate(metrics + ("all_metrics",)):
        a = values if metric == "all_metrics" else values[:, index:index + 1]
        eligible = np.isfinite(a).all(axis=1)
        supporting = eligible & (a > margin).all(axis=1)
        n, k = int(eligible.sum()), int(supporting.sum())
        output[metric] = {"n_eligible": n, "n_supporting": k, "fraction": k / n if n else float("nan"), "wilson_interval": wilson_interval(k, n, confidence)}
    return output


def analyze_file(
    scores_path: str | Path, output_path: str | Path, *, iterations: int = 5000,
    seed: int = 0, tasks: Sequence[str] | None = None,
    reference_group: str = "healthy", comparison_group: str = "facial_palsy",
) -> dict:
    """Analyze private prediction JSON and write a private local JSON report.

    Schema: ``{'schema_version': 1, 'methods': [...], 'records': [...]}``.
    Every take record supplies sample_id, participant_id, group, visit_index,
    task, and a scores mapping with exactly the declared methods. Participant
    order is first appearance. Task order is explicit or first appearance.
    Extra input fields are ignored and never forwarded to the report.
    """
    source, destination = Path(scores_path).resolve(), Path(output_path).resolve()
    if source == destination:
        raise ValueError("Output must differ from the private score input")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Expected score JSON schema_version 1")
    methods = tuple(payload["methods"])
    if not methods or len(set(methods)) != len(methods) or any(not isinstance(m, str) or not m for m in methods):
        raise ValueError("methods must be a nonempty list of unique strings")
    takes, sample_ids, task_order = defaultdict(list), set(), []
    for row in payload["records"]:
        sample = row["sample_id"]
        if not isinstance(sample, str) or not sample or sample in sample_ids:
            raise ValueError("Each take requires a unique nonempty sample_id")
        sample_ids.add(sample)
        if set(row["scores"]) != set(methods):
            raise ValueError("Every take must contain exactly the declared methods")
        key = (row["participant_id"], row["group"], row["visit_index"], row["task"])
        takes[key].append([row["scores"][m] for m in methods])
        if row["task"] not in task_order:
            task_order.append(row["task"])
    tasks = tuple(task_order if tasks is None else tasks)
    records = [ScoreRecord(p, g, v, t, tuple(aggregate_take_scores(a))) for (p, g, v, t), a in takes.items()]
    auc = cluster_auc(records, methods, tasks, reference_group=reference_group, comparison_group=comparison_group, n_bootstrap=iterations, seed=seed)
    paired = []
    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:
            paired.append({"method_a": method_a, "method_b": method_b, **auc.paired_difference(method_a, method_b)})
    comparison_records = [r for r in records if r.group == comparison_group]
    visit_tasks = defaultdict(set)
    for row in comparison_records:
        visit_tasks[(row.participant_id, row.visit_index)].add(row.task)
    if all(values == set(tasks) for values in visit_tasks.values()):
        followup = {"status": "complete_task_visits", **summarize_followup(comparison_records, methods, tasks)}
    else:
        followup = {"status": "unavailable", "reason": "At least one comparison visit lacks a declared task; no changing-task global means or endpoints were calculated."}
    variability = within_visit_variability({(p, v, t): np.asarray(a, float) for (p, g, v, t), a in takes.items() if g == comparison_group})
    result = {
        "schema_version": 1,
        "scope": "Private frozen-score analysis; no model refitting, clinical inference, or elapsed-time interpretation",
        "methods": methods, "tasks": auc.tasks, "policies": auc.policies,
        "bootstrap": {"iterations": iterations, "seed": seed, "confidence": auc.confidence, "unit": "participant clusters stratified by group", "shared_across": "all methods, tasks, and aggregation policies", "order": "participant first appearance in input; declared task/method order"},
        "auc_estimate": auc.estimate,
        "auc_interval": auc.interval() if iterations else None,
        "paired_method_differences": paired,
        "score_omission": leave_one_subject_out(records, methods, tasks, reference_group=reference_group, comparison_group=comparison_group),
        "ordinal_followup": followup,
        "within_visit_take_variability": variability,
        "regional_patterns": {"status": "not_computed_from_scores", "reason": "Use regional_summary and side_contrasts on private maps with explicit local region/side assignments."},
    }

    def serializable(value):
        if isinstance(value, np.ndarray):
            return serializable(value.tolist())
        if isinstance(value, np.generic):
            return serializable(value.item())
        if isinstance(value, dict):
            return {key: serializable(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    result = serializable(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
