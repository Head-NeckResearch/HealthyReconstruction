"""Crop and resample facial motion without redefining the resting baseline.

This module does not *select* rest. A quality-checked raw-frame index must be
provided by a separate selection/review policy. ``N`` is an exact copy of that
frame, even when the frame falls outside the analysis crop. Consequently the
first displacement is generally nonzero: ``X[0] == 0`` is not an invariant.

The time coordinate is a zero-based row index in a consecutive raw recording,
not the CSV's frame identifier. Check acquisition-time/frame-ID discontinuities
upstream. All interpolation uses complete XYZ observations of a marker and
support points inside the half-open crop; there is no endpoint extrapolation.

Observation support is deliberately separate from model-input support. Values
interpolated between two *adjacent observed* raw rows are bracket-supported,
whereas interpolation across missing rows is explicitly imputed. Neither is
described as an original measured sample when the time coordinate is fractional.
The maximum gap is the number of missing integer raw rows between support
points. Its default is an engineering limit, not a data-fitted clinical rule.

No source data, existing pipeline, or frozen model/output files are modified.
"""

from __future__ import annotations

import operator

import numpy as np


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _frames_array(aligned_frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(aligned_frames)
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ValueError("aligned_frames must have shape (raw_frames, markers, 3)")
    if not frames.shape[0] or not frames.shape[1]:
        raise ValueError("aligned_frames must contain at least one frame and marker")
    if frames.dtype.kind not in "fiu":
        raise ValueError("aligned_frames must contain real numeric coordinates")
    return frames


def displacement_energy_from_neutral(
    aligned_frames: np.ndarray, neutral: np.ndarray
) -> np.ndarray:
    """Mean Euclidean displacement over jointly observed full-XYZ markers.

    ``neutral`` is an explicitly supplied pose, not implicitly the first raw or
    cropped frame. Unsupported frames receive NaN, not zero. Compute this on
    the uncropped sequence when using its result to select an active window.
    """
    frames = _frames_array(aligned_frames)
    baseline = np.asarray(neutral)
    if baseline.shape != frames.shape[1:] or baseline.dtype.kind not in "fiu":
        raise ValueError("neutral must be a real numeric array of shape (markers, 3)")
    valid = np.isfinite(frames).all(axis=-1) & np.isfinite(baseline).all(axis=-1)
    distances = np.zeros(valid.shape, dtype=np.float64)
    frame_id, marker_id = np.nonzero(valid)
    distances[frame_id, marker_id] = np.linalg.norm(
        frames[frame_id, marker_id].astype(np.float64)
        - baseline[marker_id].astype(np.float64), axis=-1
    )
    counts = valid.sum(axis=-1)
    result = np.full(len(frames), np.nan, dtype=np.float64)
    np.divide(distances.sum(axis=-1), counts, out=result, where=counts > 0)
    return result


def anchored_representation(
    aligned_frames: np.ndarray,
    neutral_raw_index: int,
    crop_start: int,
    crop_stop: int,
    n_frames: int = 100,
    max_interp_gap_frames: int = 5,
) -> dict[str, np.ndarray]:
    """Build a neutral-anchored representation with explicit validity provenance.

    Args:
        aligned_frames: Already dental-stabilized/aligned coordinates, normally
            ``(T, 105, 3)``. Other marker counts are accepted for reuse/testing.
        neutral_raw_index: Independently selected rest frame in the full raw
            sequence. It need not belong to the analysis crop.
        crop_start, crop_stop: Raw row bounds, inclusive/exclusive respectively.
        n_frames: Number of equally spaced output times, including both crop
            endpoints when > 1. A single output time is the crop's first row.
        max_interp_gap_frames: Largest number of missing *raw rows* that can be
            bridged for model input only. Zero disables gap filling. A gap of
            five missing rows has observed support endpoints six rows apart.

    Returns:
        N: Exact selected raw pose, with original missing coordinates preserved.
        P: Resampled positions; an unsupported marker has three NaNs.
        X: ``P - N`` where ``B_input`` holds, zero elsewhere.
        B_neutral: Full-XYZ observation validity of the selected raw pose.
        B_raw_observed: Full-XYZ observation validity for every original raw row.
        B_observed: Exact observed raw point OR fractional time bracketed by two
            adjacent fully observed rows. Independent of neutral validity.
        B_imputed: Positions supported only by bounded interpolation over one
            or more missing raw rows. Disjoint from ``B_observed``.
        B_input: Position support (observed OR imputed) AND valid neutral.
        B_target: Observed bracket support AND valid neutral; use for movement
            targets/scoring. Imputed points are never observation-supported.
        raw_positions_resampled: Fractional zero-based raw row coordinates.
        raw_support_left, raw_support_right: Original observed support row for
            each resampled marker, or -1 when unsupported. Exact observed points
            have equal support endpoints.

    All returned arrays own their storage. Neither this function nor the energy
    helper alters the input. There is no reversal, forced-zero initial motion,
    neutral interpolation, long-gap bridging, or endpoint extrapolation.
    """
    frames = _frames_array(aligned_frames)
    total_frames, n_markers, _ = frames.shape
    neutral_index = _integer(neutral_raw_index, "neutral_raw_index")
    start = _integer(crop_start, "crop_start")
    stop = _integer(crop_stop, "crop_stop", minimum=1)
    count = _integer(n_frames, "n_frames", minimum=1)
    max_gap = _integer(max_interp_gap_frames, "max_interp_gap_frames")
    if neutral_index >= total_frames:
        raise ValueError("neutral_raw_index is outside the raw recording")
    if not 0 <= start < stop <= total_frames:
        raise ValueError("crop must satisfy 0 <= crop_start < crop_stop <= raw length")

    neutral = frames[neutral_index].copy()
    raw_valid = np.isfinite(frames).all(axis=-1)
    neutral_valid = raw_valid[neutral_index].copy()
    positions = np.linspace(start, stop - 1, count, dtype=np.float64)
    # Snap roundoff around mathematically integral times, never arbitrary times.
    integral = np.rint(positions)
    positions[np.abs(positions - integral) <= 1e-12] = integral[
        np.abs(positions - integral) <= 1e-12
    ]

    points = np.full((count, n_markers, 3), np.nan, dtype=np.float64)
    observed = np.zeros((count, n_markers), dtype=bool)
    imputed = np.zeros_like(observed)
    support_left = np.full(observed.shape, -1, dtype=np.int64)
    support_right = np.full(observed.shape, -1, dtype=np.int64)

    for marker in range(n_markers):
        observed_rows = np.flatnonzero(raw_valid[start:stop, marker]) + start
        if not len(observed_rows):
            continue
        right_indices = np.searchsorted(observed_rows, positions, side="left")
        bounded_right = np.minimum(right_indices, len(observed_rows) - 1)
        exact = (right_indices < len(observed_rows)) & (
            observed_rows[bounded_right] == positions
        )
        if exact.any():
            exact_rows = observed_rows[bounded_right[exact]]
            points[exact, marker] = frames[exact_rows, marker]
            observed[exact, marker] = True
            support_left[exact, marker] = exact_rows
            support_right[exact, marker] = exact_rows

        bracket = (~exact) & (right_indices > 0) & (right_indices < len(observed_rows))
        target_rows = np.flatnonzero(bracket)
        if not len(target_rows):
            continue
        left = observed_rows[right_indices[target_rows] - 1]
        right = observed_rows[right_indices[target_rows]]
        missing_rows = right - left - 1
        admissible = missing_rows <= max_gap
        target_rows, left, right, missing_rows = (
            array[admissible] for array in (target_rows, left, right, missing_rows)
        )
        if not len(target_rows):
            continue
        alpha = (positions[target_rows] - left) / (right - left)
        points[target_rows, marker] = (
            (1 - alpha[:, None]) * frames[left, marker]
            + alpha[:, None] * frames[right, marker]
        )
        observed[target_rows, marker] = missing_rows == 0
        imputed[target_rows, marker] = missing_rows > 0
        support_left[target_rows, marker] = left
        support_right[target_rows, marker] = right

    input_valid = (observed | imputed) & neutral_valid[None, :]
    target_valid = observed & neutral_valid[None, :]
    displacement = np.zeros_like(points)
    time_id, marker_id = np.nonzero(input_valid)
    displacement[time_id, marker_id] = (
        points[time_id, marker_id] - neutral[marker_id]
    )
    return {
        "N": neutral,
        "X": displacement,
        "P": points,
        "B_neutral": neutral_valid,
        "B_raw_observed": raw_valid,
        "B_input": input_valid,
        "B_observed": observed,
        "B_imputed": imputed,
        "B_target": target_valid,
        "raw_positions_resampled": positions,
        "raw_support_left": support_left,
        "raw_support_right": support_right,
    }
