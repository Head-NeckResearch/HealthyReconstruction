"""Unblended healthy references, common-support filtering, and geometric maps."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from .settings import DIRECTION_FLOOR_MM, validate_direction_rule
EPS = 1e-12
def interpolate_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size)
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return np.full_like(y, np.nan, dtype=float)
    if ok.sum() == 1:
        return np.full_like(y, float(y[ok][0]), dtype=float)
    return np.interp(x, x[ok], y[ok]).astype(float)


def moving_average_1d(y: np.ndarray, window: int = 7) -> np.ndarray:
    y_interp = interpolate_nan_1d(y)
    if not np.isfinite(y_interp).any():
        return y_interp
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if y_interp.size < window:
        return y_interp
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(y_interp, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def lowpass_1d(y: np.ndarray, fs_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    y_interp = interpolate_nan_1d(y)
    if not np.isfinite(y_interp).any():
        return y_interp
    if cutoff_hz <= 0 or cutoff_hz >= fs_hz / 2:
        return y_interp
    if butter is not None and filtfilt is not None:
        try:
            b, a = butter(int(order), cutoff_hz / (fs_hz / 2.0), btype="low")
            padlen = 3 * max(len(a), len(b))
            if y_interp.size > padlen:
                return filtfilt(b, a, y_interp)
        except Exception:
            pass
    if savgol_filter is not None:
        try:
            win = max(5, int(round(fs_hz / max(cutoff_hz, EPS))))
            if win % 2 == 0:
                win += 1
            win = min(win, y_interp.size if y_interp.size % 2 == 1 else y_interp.size - 1)
            if win >= 5:
                return savgol_filter(y_interp, window_length=win, polyorder=2, mode="interp")
        except Exception:
            pass
    return moving_average_1d(y_interp, window=7)


def filter_sequence(seq: np.ndarray, valid_mask: np.ndarray, fs_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Filter trajectory coordinates marker-wise and dimension-wise.

    Invalid samples are interpolated only to stabilize the filter, then restored to NaN.
    """
    seq = np.asarray(seq, dtype=float)
    T, M, C = seq.shape
    out = np.full_like(seq, np.nan, dtype=float)
    for m in range(M):
        marker_valid = valid_mask[:, m] & np.isfinite(seq[:, m, :]).all(axis=1)
        if marker_valid.sum() < 2:
            continue
        for c in range(C):
            y = seq[:, m, c].astype(float).copy()
            y[~marker_valid] = np.nan
            yf = lowpass_1d(y, fs_hz=fs_hz, cutoff_hz=cutoff_hz, order=order)
            yf[~marker_valid] = np.nan
            out[:, m, c] = yf
    return out

def finite_mean(values: np.ndarray, axis=None) -> np.ndarray:
    """Mean of finite values without converting missing values to zeros."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def maps_from_filtered(
    observed: np.ndarray, reference: np.ndarray,
    observed_neutral: np.ndarray, reference_neutral: np.ndarray,
    validity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate AE descriptors with the fixed 1-mm amplitude eligibility rule.

    Both excursion amplitudes must reach 1 mm for counter-direction values.
    Ineligible and unsupported values remain missing; eligible zeros are kept.
    """
    observed, reference = np.asarray(observed, float), np.asarray(reference, float)
    valid = np.asarray(validity, bool) & np.isfinite(observed).all(-1) & np.isfinite(reference).all(-1)
    raw = np.linalg.norm(observed - reference, axis=-1)
    raw[~valid] = np.nan
    x = observed - np.asarray(observed_neutral, float)[None]
    h = reference - np.asarray(reference_neutral, float)[None]
    motion_valid = valid & np.isfinite(x).all(-1) & np.isfinite(h).all(-1)
    a, b = np.linalg.norm(x, axis=-1), np.linalg.norm(h, axis=-1)
    excess, reduced = np.maximum(a - b, 0.0), np.maximum(b - a, 0.0)
    excess[~motion_valid], reduced[~motion_valid] = np.nan, np.nan
    directional_valid = motion_valid & (a >= DIRECTION_FLOOR_MM) & (b >= DIRECTION_FLOOR_MM)
    opposite = np.divide(-np.sum(x * h, axis=-1), b, out=np.full_like(a, np.nan), where=directional_valid)
    return {
        "raw_error": raw, "excess_movement": excess,
        "reduced_movement": reduced, "counter_direction": np.maximum(opposite, 0.0),
    }

METHODS = ('ae','without_static_projection','without_temporal_projection',
           'healthy_mean_trajectory','target_neutral_plus_healthy_displacement')


def prediction_variants(neutral, static, motion, observed_motion, reference_neutral, reference_motion):
    """Final article variants; all neural outputs are raw unblended decodes."""
    return {
        'ae':(static[None]+motion,static),
        'without_static_projection':(neutral[None]+motion,neutral),
        'without_temporal_projection':(static[None]+observed_motion,static),
        'healthy_mean_trajectory':(reference_neutral[None]+reference_motion,reference_neutral),
        'target_neutral_plus_healthy_displacement':(neutral[None]+reference_motion,neutral),
    }


def common_scoring_mask(target, mask, predictions):
    """Identical whole-XYZ support across methods, with >=2 times per marker."""
    valid=np.asarray(mask,bool)&np.isfinite(target).all(-1)
    for positions,_ in predictions.values():
        valid &= np.isfinite(positions).all(-1)
    valid &= valid.sum(axis=0,keepdims=True)>=2
    return valid


def _load_model(path,device):
    import torch
    from .models import StaticNeutralAE, DynamicMotionProjector
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    model_type=StaticNeutralAE if checkpoint['branch']=='static' else DynamicMotionProjector
    model=model_type(**checkpoint['parameters']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model,checkpoint


def _decode(model,checkpoint,values,mask,device):
    import torch
    from .models import normalize
    mean,std=np.asarray(checkpoint['mean'],np.float32),np.asarray(checkpoint['std'],np.float32)
    z=normalize(values,mask,mean,std)
    with torch.no_grad():
        decoded=model(torch.from_numpy(z[None]).to(device),torch.from_numpy(mask[None].astype(np.float32)).to(device)).cpu().numpy()[0]
    return (decoded*std+mean).astype(np.float32)


def predict(dataset_root,models_root,output_root,config,device='cpu'):
    """Write five cohort-comparison scores and AE-only regional maps privately."""
    import torch
    from .data import DEVELOPMENT, load_prepared, load_sample, write_json
    from .train import file_hash
    validate_direction_rule(config)
    dataset_root,models_root,output_root=Path(dataset_root),Path(models_root),Path(output_root)
    if output_root.exists():
        raise FileExistsError('Prediction output already exists; choose a fresh private directory')
    protocol=json.loads((models_root/'protocol.json').read_text())
    if config!=protocol['config']:
        raise ValueError('Prediction configuration differs from frozen training configuration')
    manifest=load_prepared(dataset_root)
    # Check frozen development content and references before inference.
    dev={r['sample_id']:file_hash(dataset_root/r['file']) for r in manifest['records'] if r['role'] in DEVELOPMENT}
    refs={task:file_hash(dataset_root/'references'/f'{task}.npz') for task in manifest['tasks']}
    if dev!=protocol['development_input_hashes'] or refs!=protocol['reference_hashes']:
        raise ValueError('Development inputs or healthy references changed after training')
    device=torch.device(device)
    static,static_checkpoint=_load_model(models_root/'static'/'model.pt',device)
    dynamic={task:_load_model(models_root/'dynamic'/task/'model.pt',device) for task in manifest['tasks']}
    references={}
    for task in manifest['tasks']:
        with np.load(dataset_root/'references'/f'{task}.npz',allow_pickle=False) as z:
            references[task]=(z['N_ref'],z['X_ref'])
    chosen=[r for r in manifest['records'] if r['role'] not in DEVELOPMENT]
    if not chosen:
        raise ValueError('No external evaluation records in prepared manifest')
    output_root.mkdir(parents=True); (output_root/'predictions').mkdir()
    scores=[]; ec=config['evaluation']
    for r in chosen:
        a=load_sample(dataset_root,r)
        neutral_mask=np.isfinite(a['N']).all(-1)
        static_prediction=_decode(static,static_checkpoint,a['N'],neutral_mask,device)
        model,checkpoint=dynamic[r['task']]
        motion=_decode(model,checkpoint,a['X'],a['M'].astype(bool),device)
        nr,xr=references[r['task']]
        variants=prediction_variants(a['N'],static_prediction,motion,a['X'],nr,xr)
        target=a['N'][None]+a['X']
        support=common_scoring_mask(target,a['M'],variants)
        if not support.any():
            raise ValueError('Evaluation sample has no common supported coordinates')
        observed=filter_sequence(target,support,ec['fs_hz'],ec['lowpass_hz'],ec['filter_order'])
        values={}; saved={}
        for method,(positions,own_neutral) in variants.items():
            filtered=filter_sequence(positions,support,ec['fs_hz'],ec['lowpass_hz'],ec['filter_order'])
            valid=support & np.isfinite(observed).all(-1) & np.isfinite(filtered).all(-1)
            raw_error=np.linalg.norm(observed-filtered,axis=-1)
            raw_error[~valid]=np.nan
            # Equal marker contribution after each marker's supported-time mean.
            values[method]=float(finite_mean(finite_mean(raw_error,axis=0)))
            if not np.isfinite(values[method]):
                raise ValueError('Nonfinite common-support trajectory score')
            if method=='ae':
                maps=maps_from_filtered(observed,filtered,a['N'],own_neutral,support)
                for metric,values_array in maps.items():
                    saved[f'{method}__{metric}']=values_array
        identity={k:r[k] for k in ('sample_id','participant_id','visit_index','task')}
        scores.append({**identity,'group':'healthy' if r['role']=='healthy_eval' else 'facial_palsy','scores':values})
        np.savez_compressed(output_root/'predictions'/f"{r['sample_id']}.npz",N=a['N'],N_AE=static_prediction,X_AE=motion,
                            N_ref=nr,X_ref=xr,X=a['X'],P_target=target,B_scoring=support,**saved)
    write_json(output_root/'scores.json',{'schema_version':1,'methods':list(METHODS),'records':scores})
    write_json(output_root/'protocol.json',{'schema_version':1,'config':config,'device':str(device),
        'methods':list(METHODS),'primary_reference':'N_AE + X_AE, no blending',
        'regional_map_methods':['ae'],
        'score':'Mean over markers of supported-time mean Euclidean absolute trajectory error',
        'filter_domain':'100-point normalized movement phase, cutoff 0.08 cycles per normalized sample'})
    return scores
