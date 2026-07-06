#!/usr/bin/env python3
"""Merge braid 3D rows with strand-cam flytrax 2D detections for LED calibration.

Used by collect_led_calibration_data.py at the end of a recording session, and
importable by the later calibration script for re-loading datasets. Standalone
usage:

    python3 led_calibration_merge.py --selftest
    python3 led_calibration_merge.py DATASET_DIR        # re-run merge on a dataset

Clock domains: braid rows carry trigger_timestamp (triggerbox clock, unix
based, from the braid PC); flytrax rows carry created_at + time_microseconds
(this PC's clock). The two PCs' clocks may disagree. Both cameras are driven
by the same hardware triggerbox, so the frame RATE is identical and the clock
offset is constant over a session. The offset is estimated by cross-correlating
the LED's speed profile seen in 3D and in 2D, then rows are matched by nearest
corrected timestamp, and finally re-matched exactly by the (constant) integer
frame-number offset when it is consistent.
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.signal

DEFAULT_PARAMS = {
    'match_tolerance_frames': 0.5,   # nearest-time acceptance, in frame periods
    'max_offset_search_s': 10.0,     # correlation lag search half-window around the prior
    'min_objid_rows': 5,             # drop spurious braid obj_ids shorter than this
    'gap_nan_s': 0.5,                # speed-series gaps longer than this become NaN
    'max_series_s': 900.0,           # cap correlation series length
    'frame_lock_min_pairs': 20,      # min time-matched pairs to attempt frame locking
    'frame_lock_consistency': 0.95,  # min fraction agreeing on the frame offset
}

MERGED_COLUMNS = ['frame_braid', 'frame_flytrax', 'trigger_timestamp',
                  't_flytrax_corrected', 'residual_ms', 'obj_id',
                  'x', 'y', 'z', 'xvel', 'yvel', 'zvel', 'P00', 'P11', 'P22',
                  'x_px', 'y_px', 'central_moment', 'orientation_radians_mod_pi']


# --------------------------------------------------------------------- loading

def _find_created_at(obj):
    '''Recursively find a "created_at" value in the flytrax header JSON.'''
    if isinstance(obj, dict):
        if 'created_at' in obj:
            return obj['created_at']
        for v in obj.values():
            found = _find_created_at(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_created_at(v)
            if found is not None:
                return found
    return None


def _parse_time(value):
    '''Parse a created_at value (ISO8601 string or unix float) to unix seconds.'''
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace('Z', '+00:00')
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc).astimezone()
    return dt.timestamp()


def load_flytrax(path):
    '''Load a strand-cam flytrax CSV. Returns (DataFrame, created_at_unix, header_json).

    The DataFrame has columns t2d (unix, local clock), frame_flytrax, x_px,
    y_px, central_moment, orientation_radians_mod_pi, one row per frame with a
    detection.
    '''
    # strand-cam >= 1.0 writes the header as '#'-commented YAML between
    # '-- start of yaml config --' / '-- end --' markers; older versions used
    # a JSON line. Parse either.
    header = None
    yaml_lines = []
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            stripped = line.lstrip('#').strip()
            if stripped.startswith('{'):
                try:
                    header = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
            elif not stripped.startswith('--'):
                yaml_lines.append(line.lstrip('#').rstrip('\n').removeprefix(' '))
    if header is None and yaml_lines:
        try:
            import yaml as _yaml
            header = _yaml.safe_load('\n'.join(yaml_lines))
        except Exception:
            header = None

    created_at_raw = _find_created_at(header) if header else None
    if created_at_raw is None:
        raise ValueError('%s: no created_at found in flytrax header comments' % path)
    created_at_unix = _parse_time(created_at_raw)

    df = pd.read_csv(path, comment='#')
    df = df[np.isfinite(df['x_px'])].copy()
    df['t2d'] = created_at_unix + df['time_microseconds'] * 1e-6
    # single LED expected; if multiple detections share a frame keep the largest
    if 'central_moment' in df.columns:
        df = df.sort_values(['frame', 'central_moment']).drop_duplicates('frame', keep='last')
    else:
        df = df.drop_duplicates('frame', keep='first')
        df['central_moment'] = np.nan
    if 'orientation_radians_mod_pi' not in df.columns:
        df['orientation_radians_mod_pi'] = np.nan
    df = df.rename(columns={'frame': 'frame_flytrax'}).sort_values('frame_flytrax')
    cols = ['frame_flytrax', 't2d', 'x_px', 'y_px', 'central_moment',
            'orientation_radians_mod_pi']
    return df[cols].reset_index(drop=True), created_at_unix, header


def load_braid(path):
    '''Load braid_3d.csv written by collect_led_calibration_data.py.'''
    df = pd.read_csv(path)
    return df.rename(columns={'frame': 'frame_braid'})


def select_single_target(df, min_objid_rows=5):
    '''Reduce braid rows to one row per frame, treating all obj_ids as one LED.

    Drops obj_ids with fewer than min_objid_rows rows (spurious ghosts); when
    several obj_ids share a frame, keeps the row from the obj_id with the most
    rows overall. Returns (DataFrame sorted by frame, report dict).
    '''
    sizes = df.groupby('obj_id')['frame_braid'].size()
    keep = sizes[sizes >= min_objid_rows].index
    report = {'n_objids_raw': int(len(sizes)),
              'n_objids_kept': int(len(keep)),
              'n_rows_raw': int(len(df))}
    df = df[df['obj_id'].isin(keep)].copy()
    df['_objsize'] = df['obj_id'].map(sizes)
    n_before = len(df)
    df = (df.sort_values(['frame_braid', '_objsize'], ascending=[True, False])
            .drop_duplicates('frame_braid', keep='first')
            .drop(columns='_objsize')
            .reset_index(drop=True))
    report['n_multi_frames'] = int(n_before - len(df))
    report['n_rows_kept'] = int(len(df))
    return df, report


def _fit_frame_clock(frames, t):
    '''Least-squares fit t = a + b*frame; returns (fitted_t, b).

    Frame numbers are exact integer counters driven by the triggerbox, so
    fitting the clock against them averages per-row timestamp jitter across
    the whole recording and yields effectively jitter-free timestamps.
    '''
    f = np.asarray(frames, dtype=float)
    b, a = np.polyfit(f, np.asarray(t, dtype=float), 1)
    return a + b * f, float(b)


# ---------------------------------------------------------- offset estimation

def _speed_series(t, positions, frames, max_frame_gap=2):
    '''Central-difference speed |dp/dt| at interior samples with small frame gaps.

    Finite differences on raw positions rather than braid's Kalman velocities:
    the Kalman filter lag would bias the offset estimate.
    '''
    t = np.asarray(t, dtype=float)
    p = np.asarray(positions, dtype=float)
    f = np.asarray(frames, dtype=float)
    if len(t) < 3:
        return np.array([]), np.array([])
    dp = np.linalg.norm(p[2:] - p[:-2], axis=1)
    dt = t[2:] - t[:-2]
    valid = ((f[1:-1] - f[:-2] <= max_frame_gap) &
             (f[2:] - f[1:-1] <= max_frame_gap) & (dt > 0))
    return t[1:-1][valid], (dp / np.where(dt > 0, dt, np.nan))[valid]


def _resample_zscore(t, s, t0, t1, dt, gap_nan_s):
    '''Resample (t, s) to a uniform grid [t0, t1); z-score; gaps/NaN -> 0.'''
    grid = np.arange(t0, t1, dt)
    if len(t) < 2 or len(grid) < 2:
        return grid, np.zeros(len(grid))
    vals = np.interp(grid, t, s)
    # blank grid points farther than gap_nan_s/2 from any real sample
    idx = np.searchsorted(t, grid)
    d_left = grid - t[np.clip(idx - 1, 0, len(t) - 1)]
    d_right = t[np.clip(idx, 0, len(t) - 1)] - grid
    dist = np.minimum(np.abs(d_left), np.abs(d_right))
    vals[dist > gap_nan_s / 2.0] = np.nan
    good = np.isfinite(vals)
    if good.sum() < 10 or np.nanstd(vals) == 0:
        return grid, np.zeros(len(grid))
    z = (vals - np.nanmean(vals)) / np.nanstd(vals)
    z[~good] = 0.0
    return grid, z


def estimate_offset(braid, flytrax, dt, prior, params):
    '''Estimate the clock offset delta such that t_braid = t_flytrax + delta.

    Cross-correlates z-scored LED speed profiles from both streams. Returns a
    report dict with offset_seconds, peak_correlation, second_peak_ratio.
    '''
    t3, s3 = _speed_series(braid['trigger_timestamp'].values,
                           braid[['x', 'y', 'z']].values,
                           braid['frame_braid'].values)
    t2, s2 = _speed_series(flytrax['t2d'].values,
                           flytrax[['x_px', 'y_px']].values,
                           flytrax['frame_flytrax'].values)
    if len(t3) < 10 or len(t2) < 10:
        raise ValueError('not enough movement samples to correlate '
                         '(3d: %d, 2d: %d) -- was the LED moving and visible?'
                         % (len(t3), len(t2)))

    max_len = params['max_series_s']
    g3, z3 = _resample_zscore(t3, s3, t3[0], min(t3[-1], t3[0] + max_len), dt,
                              params['gap_nan_s'])
    g2, z2 = _resample_zscore(t2, s2, t2[0], min(t2[-1], t2[0] + max_len), dt,
                              params['gap_nan_s'])

    c = scipy.signal.correlate(z3, z2, mode='full', method='fft')
    lags = np.arange(-(len(z2) - 1), len(z3))
    deltas = (g3[0] - g2[0]) + lags * dt

    mask = np.abs(deltas - prior) <= params['max_offset_search_s']
    if not mask.any():
        raise ValueError('no correlation lags within %.1f s of the clock-offset '
                         'prior (%.3f s) -- increase max_offset_search_s?'
                         % (params['max_offset_search_s'], prior))
    cm = np.where(mask, c, -np.inf)
    li = int(np.argmax(cm))

    # parabolic sub-sample refinement
    frac = 0.0
    if 0 < li < len(c) - 1:
        denom = c[li - 1] - 2 * c[li] + c[li + 1]
        if denom != 0:
            frac = float(np.clip(0.5 * (c[li - 1] - c[li + 1]) / denom, -1, 1))
    offset = float(deltas[li] + frac * dt)

    # quality: pearson r of the two series at the integer peak lag
    lag = int(lags[li])
    if lag >= 0:
        a, b = z3[lag:lag + len(z2)], z2[:max(0, len(z3) - lag)]
    else:
        a, b = z3[:len(z2) + lag], z2[-lag:-lag + len(z3)]
    n = min(len(a), len(b))
    both = (a[:n] != 0) & (b[:n] != 0)
    peak_r = float(np.corrcoef(a[:n][both], b[:n][both])[0, 1]) if both.sum() > 10 else 0.0

    # second peak at least 2 s away from the chosen one
    far = mask & (np.abs(deltas - offset) >= 2.0)
    second_ratio = float(np.max(c[far]) / c[li]) if far.any() and c[li] > 0 else 0.0

    return {'offset_seconds': offset,
            'offset_prior_seconds': float(prior),
            'peak_correlation': peak_r,
            'second_peak_ratio': second_ratio,
            'n_speed_samples_3d': int(len(t3)),
            'n_speed_samples_2d': int(len(t2))}


# ------------------------------------------------------------------- matching

def _dlt_median_error(pts3d, pts2d, max_points=2000):
    '''Median reprojection error (px) of a linear DLT fit to 3D<->2D pairs.

    Used to pick the correct integer frame offset: pairing pixels with 3D
    positions one frame off misplaces every moving point by one frame of
    motion, which no single projection matrix can absorb.
    '''
    n = len(pts3d)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        pts3d, pts2d = pts3d[idx], pts2d[idx]
        n = max_points
    # normalize for conditioning
    m3, s3 = pts3d.mean(axis=0), pts3d.std(axis=0).mean() or 1.0
    m2, s2 = pts2d.mean(axis=0), pts2d.std(axis=0).mean() or 1.0
    X = np.c_[(pts3d - m3) / s3, np.ones(n)]
    uv = (pts2d - m2) / s2
    A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = X
    A[0::2, 8:12] = -uv[:, 0:1] * X
    A[1::2, 4:8] = X
    A[1::2, 8:12] = -uv[:, 1:2] * X
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    P = vt[-1].reshape(3, 4)
    proj = X @ P.T
    w = np.where(np.abs(proj[:, 2]) > 1e-12, proj[:, 2], 1e-12)
    err = np.linalg.norm(proj[:, :2] / w[:, None] - uv, axis=1) * s2
    return float(np.median(err))

def match_rows(braid, flytrax, offset, dt, params, frame_lock_ok=True):
    '''Match flytrax rows to braid rows. Returns (merged DataFrame, report).

    Stage 1: nearest corrected timestamp within tolerance. Stage 2: if the
    frame-number offset k = frame_braid - frame_flytrax is consistent across
    stage-1 pairs, re-match exactly by k (hardware-synced counters).
    '''
    fly = flytrax.copy()
    fly['t_flytrax_corrected'] = fly['t2d'] + offset
    fly = fly.sort_values('t_flytrax_corrected')
    br = braid.sort_values('trigger_timestamp')
    tol = params['match_tolerance_frames'] * dt

    stage1 = pd.merge_asof(fly, br, left_on='t_flytrax_corrected',
                           right_on='trigger_timestamp',
                           direction='nearest', tolerance=tol)
    stage1 = stage1[np.isfinite(stage1['trigger_timestamp'])].copy()
    report = {'n_time_matched': int(len(stage1)), 'match_method': 'time',
              'frame_offset': None, 'frame_offset_consistency': 0.0,
              'frame_offset_dlt_errors_px': {}, 'frame_offset_dlt_margin': None}

    merged = stage1
    if frame_lock_ok and len(stage1) >= params['frame_lock_min_pairs']:
        k = (stage1['frame_braid'] - stage1['frame_flytrax']).astype(np.int64)
        counts = k.value_counts()
        k0 = int(counts.index[0])
        report['frame_offset_consistency'] = float(counts.iloc[0] / len(k))

        # the correlation offset can be off by a frame or two (perspective
        # distorts the 2D speed profile), so scan candidate frame offsets
        # around k0 and pick the one whose implied 3D<->2D pairing is
        # geometrically consistent (lowest DLT reprojection error)
        dlt_errors = {}
        candidates = {}
        for k_try in range(k0 - 3, k0 + 4):
            fly2 = fly.copy()
            fly2['frame_braid'] = fly2['frame_flytrax'] + k_try
            cand = fly2.merge(br, on='frame_braid', how='inner')
            if len(cand) < params['frame_lock_min_pairs']:
                continue
            dlt_errors[k_try] = _dlt_median_error(cand[['x', 'y', 'z']].values,
                                                  cand[['x_px', 'y_px']].values)
            candidates[k_try] = cand
        if dlt_errors:
            k_hat = min(dlt_errors, key=dlt_errors.get)
            errs_sorted = sorted(dlt_errors.values())
            merged = candidates[k_hat]
            report['frame_offset'] = k_hat
            report['frame_offset_dlt_errors_px'] = {int(kk): round(v, 3)
                                                    for kk, v in dlt_errors.items()}
            report['frame_offset_dlt_margin'] = (float(errs_sorted[1] / errs_sorted[0])
                                                 if len(errs_sorted) > 1
                                                 and errs_sorted[0] > 0 else None)
            report['match_method'] = 'frame'
            # the frame lock implies the physically-consistent clock offset;
            # replace the correlation estimate with it
            offset = float((merged['trigger_timestamp'] - merged['t2d']).median())
            merged = merged.copy()
            merged['t_flytrax_corrected'] = merged['t2d'] + offset
            report['offset_seconds_frame_locked'] = offset

    merged = merged.copy()
    merged['residual_ms'] = (merged['trigger_timestamp']
                             - merged['t_flytrax_corrected']) * 1e3
    merged = merged.sort_values('frame_braid').reset_index(drop=True)
    report['n_matched'] = int(len(merged))
    report['residual_rms_ms'] = float(np.sqrt(np.mean(merged['residual_ms'] ** 2))) \
        if len(merged) else float('nan')
    return merged[[c for c in MERGED_COLUMNS if c in merged.columns]], report


# ------------------------------------------------------------------ top level

def merge_datasets(braid_csv, flytrax_csv, fps_hint=100.0, measured_fps=None,
                   params=None):
    '''Full merge: load, select target, estimate offset, match.

    Returns (merged DataFrame, report dict, extras dict for plotting).
    '''
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    flytrax, created_at, _hdr = load_flytrax(flytrax_csv)
    braid_raw = load_braid(braid_csv)
    if len(braid_raw) == 0:
        raise ValueError('no braid rows in %s' % braid_csv)
    braid, target_report = select_single_target(braid_raw, p['min_objid_rows'])
    if len(braid) < 10 or len(flytrax) < 10:
        raise ValueError('too few rows to merge (braid: %d, flytrax: %d)'
                         % (len(braid), len(flytrax)))

    # de-jitter both clocks by fitting t against the exact frame counters
    braid = braid.copy()
    flytrax = flytrax.copy()
    braid['trigger_timestamp'], dt3 = _fit_frame_clock(braid['frame_braid'],
                                                       braid['trigger_timestamp'])
    flytrax['t2d'], dt2 = _fit_frame_clock(flytrax['frame_flytrax'], flytrax['t2d'])

    # frame period: the fitted slopes are the best estimate; cross-check config.
    # Frame counters only correspond 1:1 when both cameras run off the same
    # triggerbox, i.e. when the fitted frame periods agree -- otherwise frame
    # locking must not be attempted.
    fps = measured_fps if measured_fps else fps_hint
    frame_lock_ok = abs(dt3 - dt2) / dt3 <= 0.005
    if not frame_lock_ok:
        print('WARNING: fitted frame periods disagree (braid %.6f s, flytrax '
              '%.6f s) -- are both cameras on the same triggerbox? '
              'Frame-locked matching disabled.' % (dt3, dt2))
        dt = 1.0 / fps
    else:
        dt = 0.5 * (dt3 + dt2)
    if abs(dt - 1.0 / fps) / dt > 0.02:
        print('WARNING: fitted frame period %.5f s differs from configured '
              '1/fps = %.5f s' % (dt, 1.0 / fps))

    # prior: rows generated at braid time T arrive here at local time ~(T - delta)
    # + network latency, so delta ~= trigger - receive (biased low by the latency)
    prior = float((braid['trigger_timestamp'] - braid['receive_time_unix']).median())
    offset_report = estimate_offset(braid, flytrax, dt, prior, p)
    merged, match_report = match_rows(braid, flytrax, offset_report['offset_seconds'],
                                      dt, p, frame_lock_ok=frame_lock_ok)

    report = {'fps_used': fps, 'dt_used_s': dt,
              'flytrax_created_at_unix': created_at,
              'matched_fraction_2d': float(len(merged) / len(flytrax)),
              'tolerance_frames': p['match_tolerance_frames']}
    report.update(target_report)
    report.update(offset_report)
    report.update(match_report)
    report['offset_seconds_correlation'] = offset_report['offset_seconds']
    if 'offset_seconds_frame_locked' in match_report:
        report['offset_seconds'] = match_report['offset_seconds_frame_locked']
    extras = {'braid': braid, 'flytrax': flytrax, 'params': p, 'dt': dt}
    return merged, report, extras


def plot_alignment(merged, report, extras, out_png):
    '''Diagnostic figure: aligned speed profiles, residuals, matched timeline.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    braid, flytrax = extras['braid'], extras['flytrax']
    dt, p = extras['dt'], extras['params']
    off = report['offset_seconds']

    t3, s3 = _speed_series(braid['trigger_timestamp'].values,
                           braid[['x', 'y', 'z']].values,
                           braid['frame_braid'].values)
    t2, s2 = _speed_series(flytrax['t2d'].values,
                           flytrax[['x_px', 'y_px']].values,
                           flytrax['frame_flytrax'].values)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    t_ref = t3[0] if len(t3) else 0.0

    def z(v):
        return (v - np.mean(v)) / np.std(v) if len(v) and np.std(v) > 0 else v

    axes[0].plot(t3 - t_ref, z(s3), label='3D speed (braid)', lw=0.8)
    axes[0].plot(t2 + off - t_ref, z(s2), label='2D speed (flytrax, shifted %+.3f s)' % off,
                 lw=0.8, alpha=0.75)
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('z-scored speed')
    axes[0].legend(loc='upper right')
    axes[0].set_title('aligned speed profiles  (peak r = %.2f, 2nd-peak ratio = %.2f)'
                      % (report['peak_correlation'], report['second_peak_ratio']))

    if len(merged):
        axes[1].hist(merged['residual_ms'], bins=60, color='tab:blue')
    axes[1].axvline(0, color='k', lw=0.5)
    axes[1].set_xlabel('match residual (ms)')
    axes[1].set_ylabel('count')
    axes[1].set_title('timestamp residuals  (method: %s, frame offset: %s, '
                      'consistency: %.2f, rms: %.2f ms)'
                      % (report['match_method'], report['frame_offset'],
                         report['frame_offset_consistency'],
                         report.get('residual_rms_ms', float('nan'))))

    axes[2].plot(flytrax['t2d'] + off - t_ref, np.full(len(flytrax), 0), '|',
                 ms=8, label='flytrax rows (%d)' % len(flytrax))
    axes[2].plot(braid['trigger_timestamp'] - t_ref, np.full(len(braid), 1), '|',
                 ms=8, label='braid rows (%d)' % len(braid))
    if len(merged):
        axes[2].plot(merged['trigger_timestamp'] - t_ref, np.full(len(merged), 2), '|',
                     ms=8, label='matched (%d)' % len(merged))
    axes[2].set_ylim(-0.5, 2.5)
    axes[2].set_yticks([])
    axes[2].set_xlabel('time (s)')
    axes[2].legend(loc='upper right')
    axes[2].set_title('row coverage timeline')

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_merge_on_dataset(dataset_dir, fps_hint=100.0, measured_fps=None, params=None):
    '''Locate the CSVs in a dataset dir, merge, and write merged.csv + plot.

    Returns the report dict. Used by --merge-only and by the collection script.
    '''
    braid_csv = os.path.join(dataset_dir, 'braid_3d.csv')
    flytrax_candidates = [f for f in glob.glob(os.path.join(dataset_dir, '*.csv'))
                          if os.path.basename(f) not in ('braid_3d.csv', 'merged.csv')]
    if not os.path.exists(braid_csv):
        raise FileNotFoundError('%s not found' % braid_csv)
    if len(flytrax_candidates) != 1:
        raise FileNotFoundError('expected exactly one flytrax csv in %s, found %s'
                                % (dataset_dir, flytrax_candidates))
    merged, report, extras = merge_datasets(braid_csv, flytrax_candidates[0],
                                            fps_hint, measured_fps, params)
    merged.to_csv(os.path.join(dataset_dir, 'merged.csv'), index=False)
    try:
        plot_alignment(merged, report, extras,
                       os.path.join(dataset_dir, 'alignment_diagnostics.png'))
    except Exception as err:
        print('WARNING: diagnostic plot failed: %s' % err)
    return report


# ------------------------------------------------------------------- selftest

def make_synthetic_pair(seed, tmpdir, duration_s=90.0, fps=100.0,
                        clock_offset_s=3.7, frame_offset=1234):
    '''Write synthetic braid_3d.csv + flytrax CSV with known offsets.'''
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t_grid = np.arange(n) / fps
    frames = np.arange(n) + 100

    # smooth 3D trajectory: sum of sinusoids per axis
    pos = np.zeros((n, 3))
    for ax in range(3):
        for _ in range(4):
            f = rng.uniform(0.05, 0.6)
            pos[:, ax] += rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * f * t_grid
                                                          + rng.uniform(0, 2 * np.pi))
    pos[:, 2] += 1.0  # keep in front of the camera

    # random projection to pixels
    P = rng.normal(size=(3, 4))
    P[2] = [0, 0, 1, 2.0]
    hom = np.c_[pos, np.ones(n)] @ P.T
    px = 800 * hom[:, :2] / hom[:, 2:3] + np.array([960, 600]) \
        + rng.normal(0, 0.3, size=(n, 2))

    t0 = 1_780_000_000.0
    trigger_t = t0 + t_grid + rng.normal(0, 0.002, n)          # braid clock + jitter
    t2d_true = t0 + t_grid - clock_offset_s + rng.normal(0, 0.002, n)  # local clock

    # independent visibility gaps
    vis3 = np.ones(n, bool)
    vis2 = np.ones(n, bool)
    for vis in (vis3, vis2):
        for _ in range(6):
            a = rng.integers(0, n - 300)
            vis[a:a + rng.integers(50, 300)] = False

    # braid csv: obj_id splits + one spurious short obj_id
    obj_id = np.ones(n, np.int64)
    for i, b in enumerate(np.sort(rng.integers(0, n, 3))):
        obj_id[b:] = i + 2
    braid = pd.DataFrame({'obj_id': obj_id[vis3], 'frame': frames[vis3],
                          'trigger_timestamp': trigger_t[vis3],
                          'receive_time_unix': trigger_t[vis3] - clock_offset_s
                          + 0.015 + rng.normal(0, 0.005, vis3.sum()),
                          'x': pos[vis3, 0], 'y': pos[vis3, 1], 'z': pos[vis3, 2],
                          'xvel': 0.0, 'yvel': 0.0, 'zvel': 0.0,
                          'P00': 1e-4, 'P11': 1e-4, 'P22': 1e-4,
                          'P33': 1e-4, 'P44': 1e-4, 'P55': 1e-4})
    ghost = braid.iloc[:3].copy()
    ghost['obj_id'] = 99
    ghost[['x', 'y', 'z']] += 0.5
    braid = pd.concat([braid, ghost], ignore_index=True)
    braid_path = os.path.join(tmpdir, 'braid_3d.csv')
    braid.to_csv(braid_path, index=False)

    # flytrax csv with commented header
    created_at = t2d_true[0]
    fly_frames = frames[vis2] - frame_offset
    fly_path = os.path.join(tmpdir, 'flytrax_synth.csv')
    with open(fly_path, 'w') as f:
        f.write('# flytrax synthetic selftest data\n')
        f.write('# %s\n' % json.dumps({'created_at': created_at}))
        f.write('time_microseconds,frame,central_moment,'
                'orientation_radians_mod_pi,x_px,y_px\n')
        for i, idx in enumerate(np.where(vis2)[0]):
            f.write('%d,%d,%.1f,0.0,%.3f,%.3f\n'
                    % (round((t2d_true[idx] - created_at) * 1e6), fly_frames[i],
                       100.0, px[idx, 0], px[idx, 1]))
    return braid_path, fly_path, clock_offset_s, frame_offset


def selftest():
    import tempfile
    ok = True
    for seed in (1, 2, 3):
        with tempfile.TemporaryDirectory() as tmpdir:
            braid_csv, fly_csv, true_offset, true_k = make_synthetic_pair(seed, tmpdir)
            merged, report, _extras = merge_datasets(braid_csv, fly_csv, fps_hint=100.0)
            err = abs(report['offset_seconds'] - true_offset)
            # ceiling: flytrax rows whose true partner frame is visible in braid
            fly = load_flytrax(fly_csv)[0]
            braid_frames = set(load_braid(braid_csv)['frame_braid'])
            achievable = (fly['frame_flytrax'] + true_k).isin(braid_frames).sum()
            frac = len(merged) / achievable
            passed = (err < 0.3 / 100.0
                      and report['frame_offset'] == true_k
                      and report['match_method'] == 'frame'
                      and frac > 0.98)
            ok &= passed
            print('seed %d: offset err %.4f ms, frame_offset %s (true %d), '
                  'method %s, matched %d/%d achievable (%.1f%%), peak_r %.2f -> %s'
                  % (seed, err * 1e3, report['frame_offset'], true_k,
                     report['match_method'], len(merged), achievable, 100 * frac,
                     report['peak_correlation'], 'PASS' if passed else 'FAIL'))
    print('selftest:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('dataset_dir', nargs='?',
                        help='dataset directory to (re-)merge')
    parser.add_argument('--selftest', action='store_true',
                        help='run synthetic-data self test')
    parser.add_argument('--fps', type=float, default=100.0)
    args = parser.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.dataset_dir:
        parser.error('dataset_dir required unless --selftest')
    report = run_merge_on_dataset(os.path.expanduser(args.dataset_dir),
                                  fps_hint=args.fps)
    for k in sorted(report):
        print('%s: %s' % (k, report[k]))


if __name__ == '__main__':
    main()
