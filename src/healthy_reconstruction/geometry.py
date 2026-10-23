"""Rigid dental stabilization and robust neutral registration.

Numerical routines selectively adapted from the original study implementation;
no reference coordinates or acquisition records are distributed.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
def rms(x: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        return float(np.sqrt(np.nanmean(np.square(x))))


def trimmed_indices(residuals: np.ndarray, trim_frac: float) -> np.ndarray:
    """Return boolean mask of inliers as best (1-trim_frac) residuals. residuals shape (N,)."""
    n = residuals.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(round((1.0 - trim_frac) * n)))
    order = np.argsort(residuals)
    inl = np.zeros((n,), dtype=bool)
    inl[order[:k]] = True
    return inl


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ], dtype=float)


def fixed_rot_xyz(deg_xyz: Tuple[float, float, float]) -> np.ndarray:
    """Rotation matrix for successive rotations around X then Y then Z (degrees)."""
    rx, ry, rz = [math.radians(d) for d in deg_xyz]
    Rx = rotation_matrix_from_axis_angle(np.array([1,0,0]), rx)
    Ry = rotation_matrix_from_axis_angle(np.array([0,1,0]), ry)
    Rz = rotation_matrix_from_axis_angle(np.array([0,0,1]), rz)
    return Rz @ Ry @ Rx  # common convention: apply X then Y then Z to column vectors


@dataclass
class DentalFrame:
    R: np.ndarray  # 3x3 rotation matrix
    o: np.ndarray  # 3 translation origin


def compute_dental_frame(d: np.ndarray) -> Optional[DentalFrame]:
    """
    Compute a right-handed orthonormal frame from 3 dental markers d shape (3,3).
    Returns None if degenerate / NaN.
    """
    if d.shape != (3,3):
        raise ValueError("dental markers must be (3,3)")
    if not np.isfinite(d).all():
        return None
    o = d.mean(axis=0)

    x = d[1] - d[0]
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        return None
    x = x / nx

    v = d[2] - d[0]
    # normal to dental plane
    z = np.cross(x, v)
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        return None
    z = z / nz

    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if ny < 1e-8:
        return None
    y = y / ny

    R = np.stack([x, y, z], axis=1)  # columns are basis vectors in world coords
    # Ensure R is proper rotation
    if np.linalg.det(R) < 0:
        # Flip z to make det positive
        R[:, 2] *= -1.0
    return DentalFrame(R=R, o=o)


def world_to_dental(frames_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert each frame to dental coordinates using per-frame dental frame.
    Returns:
      frames_dental: (T,108,3) in dental coords
      ok_mask: (T,) whether dental frame was valid
    """
    T = frames_world.shape[0]
    out = np.full_like(frames_world, np.nan, dtype=float)
    ok = np.zeros((T,), dtype=bool)
    for t in range(T):
        d = frames_world[t, 0:3, :]
        fr = compute_dental_frame(d)
        if fr is None:
            continue
        ok[t] = True
        # P_d = R^T (P - o)
        out[t] = (fr.R.T @ (frames_world[t] - fr.o).T).T
    return out, ok


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rigid transform (R,t) s.t. R P + t ~= Q, using Kabsch.
    P,Q: Nx3 with finite values.
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    X = P - Pc
    Y = Q - Qc
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Qc - R @ Pc
    return R, t


def apply_rt(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t


def trimmed_ransac_rigid(P: np.ndarray, Q: np.ndarray, trials: int, subset: int, trim_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Robust rigid transform using trimmed-consensus RANSAC.
    Returns R,t,inlier_mask,trimmed_rms
    """
    n = P.shape[0]
    if n < max(3, subset):
        raise ValueError(f"Not enough correspondences for RANSAC: {n}")
    best_score = float("inf")
    best_R = None
    best_t = None
    best_inl = None

    idx_all = np.arange(n)
    for _ in range(trials):
        idx = rng.choice(idx_all, size=subset, replace=False)
        try:
            R, t = kabsch(P[idx], Q[idx])
        except np.linalg.LinAlgError:
            continue
        transformed_points = apply_rt(P, R, t)
        res = np.linalg.norm(transformed_points - Q, axis=1)
        inl = trimmed_indices(res, trim_frac)
        score = rms(res[inl])
        if score < best_score:
            best_score = score
            best_R, best_t, best_inl = R, t, inl

    if best_R is None:
        raise ValueError("RANSAC failed to produce a valid transform.")

    # Refit on inliers
    R, t = kabsch(P[best_inl], Q[best_inl])
    transformed_points = apply_rt(P, R, t)
    res = np.linalg.norm(transformed_points - Q, axis=1)
    inl = trimmed_indices(res, trim_frac)
    score = rms(res[inl])
    return R, t, inl, score


def huber_irls_refine(P: np.ndarray, Q: np.ndarray, R0: np.ndarray, t0: np.ndarray, inl_mask: np.ndarray, iters: int, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple IRLS-like refinement: iteratively reweight points by Huber weights and solve weighted Kabsch
    on inliers only. Guard against NaNs and SVD non-convergence.
    """
    R, t = R0.copy(), t0.copy()
    P_inl = P[inl_mask]
    Q_inl = Q[inl_mask]
    if P_inl.shape[0] < 3:
        return R, t

    w = np.ones((P_inl.shape[0],), dtype=float)
    for _ in range(iters):
        transformed_points = apply_rt(P_inl, R, t)
        r = np.linalg.norm(transformed_points - Q_inl, axis=1)
        # Huber weights
        w = np.where(r <= k, 1.0, (k / (r + 1e-12)))
        if not np.isfinite(w).all():
            break
        # weighted centroids
        sw = np.sum(w)
        if sw < 1e-12:
            break
        Pc = np.sum(P_inl * w[:, None], axis=0) / sw
        Qc = np.sum(Q_inl * w[:, None], axis=0) / sw
        X = P_inl - Pc
        Y = Q_inl - Qc
        H = (X * w[:, None]).T @ Y
        if not np.isfinite(H).all():
            break
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            break
        Rn = Vt.T @ U.T
        if np.linalg.det(Rn) < 0:
            Vt[-1, :] *= -1
            Rn = Vt.T @ U.T
        tn = Qc - Rn @ Pc
        R, t = Rn, tn
    return R, t


def maybe_yaw_flip(P: np.ndarray, Q: np.ndarray, R: np.ndarray, t: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    """Try a 180° flip about one axis in target space before applying R,t; keep best by RMS."""
    axis = axis.upper()
    if axis not in ("X","Y","Z"):
        return R, t, False, float("inf")

    Rflip = np.eye(3)
    ax = {"X": np.array([1,0,0]), "Y": np.array([0,1,0]), "Z": np.array([0,0,1])}[axis]
    Rflip = rotation_matrix_from_axis_angle(ax, math.pi)

    # Option A: no flip
    resA = np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)
    scoreA = rms(resA)

    # Option B: flip P first
    Pf = (Rflip @ P.T).T
    resB = np.linalg.norm(apply_rt(Pf, R, t) - Q, axis=1)
    scoreB = rms(resB)

    if scoreB < scoreA:
        # absorb flip into R: R*(Rflip*P) = (R*Rflip)P
        return (R @ Rflip), t, True, scoreB
    return R, t, False, scoreA

def active_window_from_energy(
    energy: np.ndarray,
    percentile: float = 5.0,
    min_length: int = 70,
    max_inactive_gap: int = 5,
) -> tuple[int, int]:
    """Retain the existing energy-window heuristic with an explicit baseline.

    Equivalent to the previous pipeline's finite-energy window routine: join
    short inactive gaps, select the segment containing the displacement-energy
    maximum, and expand to the minimum length. This chooses an analysis window,
    not a resting frame or a clinician-validated expression peak.
    """
    values = np.asarray(energy, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("energy must be a nonempty one-dimensional array")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if isinstance(min_length, bool) or not isinstance(min_length, (int, np.integer)) or min_length < 1:
        raise ValueError("min_length must be a positive integer")
    if isinstance(max_inactive_gap, bool) or not isinstance(max_inactive_gap, (int, np.integer)) or max_inactive_gap < 0:
        raise ValueError("max_inactive_gap must be a nonnegative integer")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("No jointly observed neutral-relative movement is available")
    active = finite & (values >= np.nanpercentile(values, percentile))
    indices = np.flatnonzero(active)
    for left, right in zip(indices[:-1], indices[1:]):
        if 1 < right - left <= max_inactive_gap + 1:
            active[left:right + 1] = True
    indices = np.flatnonzero(active)
    boundaries = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[boundaries + 1]]
    stops = np.r_[indices[boundaries] + 1, indices[-1] + 1]
    peak = int(np.nanargmax(values))
    segment = np.flatnonzero((starts <= peak) & (peak < stops))[0]
    start, stop = int(starts[segment]), int(stops[segment])
    if stop - start < min_length:
        extra = min_length - (stop - start)
        start = max(0, start - (extra // 2 + extra % 2))
        stop = min(len(values), stop + extra // 2)
        while stop - start < min_length and (start > 0 or stop < len(values)):
            if start > 0:
                start -= 1
            if stop - start >= min_length:
                break
            if stop < len(values):
                stop += 1
    return start, stop


def robust_registration(neutral, template, *, trials=1200, seed=7, min_points=60):
    """Fit one proper rigid transform using whole-XYZ neutral correspondences."""
    neutral, template = np.asarray(neutral, float), np.asarray(template, float)
    valid = np.isfinite(neutral).all(-1) & np.isfinite(template).all(-1)
    if valid.sum() < min_points:
        raise ValueError("Insufficient neutral/template correspondences")
    p, q = neutral[valid], template[valid]
    if np.linalg.matrix_rank(p-p.mean(0)) < 2 or np.linalg.matrix_rank(q-q.mean(0)) < 2:
        raise ValueError("Degenerate registration geometry")
    r,t,inlier,_ = trimmed_ransac_rigid(p,q,trials=trials,subset=4,trim_frac=.1,rng=np.random.default_rng(seed))
    r,t,_,_ = maybe_yaw_flip(p,q,r,t,"Z")
    r,t = huber_irls_refine(p,q,r,t,inlier,iters=3,k=1.5)
    if not np.allclose(r@r.T,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(r),1.,atol=1e-7):
        raise ValueError("Registration failed proper rotation constraint")
    return r,t
