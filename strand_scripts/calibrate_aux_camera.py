#!/usr/bin/env python3
"""Calibrate the auxiliary camera from an LED 3D<->2D correspondence dataset.

Fits intrinsics (K), lens distortion (OpenCV plumb_bob k1,k2,p1,p2[,k3]) and
extrinsics (R, t: braid world coords -> camera) from the merged.csv written by
collect_led_calibration_data.py -- single-view resectioning of the volumetric
LED trajectory, no checkerboard required. A RANSAC-initialized DLT tolerates
heavy contamination (e.g. ~45% LED-reflection detections). Accuracy target is
visualization (overlaying braid 3D tracks on the camera video), not metrology.

    python3 calibrate_aux_camera.py DATASET_DIR
    python3 calibrate_aux_camera.py --selftest

Writes camera_calibration.yaml and calibration_diagnostics.png into
DATASET_DIR. The yaml's P_linear is drop-in compatible with
braid_analysis.braid_2d_analysis.reproject; the full distortion model is used
via load_camera_calibration() + project_points() from this module.

Note: the distortion model is only constrained inside the pixel region the LED
actually covered (valid_pixel_region in the yaml); toward uncovered sensor
edges it is an extrapolation. k3 is fixed to 0 by default for this reason.
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import scipy.optimize
import cv2
import yaml

BOUND_HIT_TOL = 1e-6


# --------------------------------------------------------------------- loading

def load_merged(dataset_dir):
    df = pd.read_csv(os.path.join(dataset_dir, 'merged.csv'))
    df = df[np.isfinite(df[['x', 'y', 'z', 'x_px', 'y_px']]).all(axis=1)]
    xyz = df[['x', 'y', 'z']].values.astype(float)
    uv = df[['x_px', 'y_px']].values.astype(float)
    meta_path = os.path.join(dataset_dir, 'metadata.yaml')
    metadata = {}
    if os.path.exists(meta_path):
        metadata = yaml.safe_load(open(meta_path)) or {}
    return df, xyz, uv, metadata


def check_geometry(xyz, uv, image_size):
    if len(xyz) < 12:
        raise ValueError('only %d correspondences -- need at least 12' % len(xyz))
    if len(xyz) < 100:
        print('WARNING: only %d correspondences; expect a rough fit' % len(xyz))
    sv = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    planarity = float(sv[2] / sv[0])
    if planarity < 0.02:
        raise ValueError('3D points are nearly planar (ratio %.4f) -- single-view '
                         'calibration is degenerate; recollect moving the LED '
                         'through more depth' % planarity)
    region = {'x_min': float(uv[:, 0].min()), 'x_max': float(uv[:, 0].max()),
              'y_min': float(uv[:, 1].min()), 'y_max': float(uv[:, 1].max())}
    w, h = image_size
    coverage = ((region['x_max'] - region['x_min']) *
                (region['y_max'] - region['y_min'])) / float(w * h)
    return {'n_points': int(len(xyz)),
            'pca_singular_values': [float(v) for v in sv],
            'planarity_ratio': planarity,
            'valid_pixel_region': region,
            'pixel_coverage_fraction': float(coverage)}


# --------------------------------------------------------------- linear stages

def fit_dlt(xyz, uv):
    '''Normalized DLT -> 3x4 projection matrix (world -> pixels).'''
    n = len(xyz)
    m3, s3 = xyz.mean(axis=0), xyz.std(axis=0).mean() or 1.0
    m2, s2 = uv.mean(axis=0), uv.std(axis=0).mean() or 1.0
    X = np.c_[(xyz - m3) / s3, np.ones(n)]
    U = (uv - m2) / s2
    A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = X
    A[0::2, 8:12] = -U[:, 0:1] * X
    A[1::2, 4:8] = X
    A[1::2, 8:12] = -U[:, 1:2] * X
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    Pn = vt[-1].reshape(3, 4)
    # denormalize: uv = T2^-1 @ Pn @ T3
    T2inv = np.array([[s2, 0, m2[0]], [0, s2, m2[1]], [0, 0, 1.0]])
    T3 = np.eye(4)
    T3[:3, :3] /= s3
    T3[:3, 3] = -m3 / s3
    P = T2inv @ Pn @ T3
    # cheirality: w = P[2] @ [X,1] must be positive for points in front
    w = np.c_[xyz, np.ones(n)] @ P[2]
    if np.median(w) < 0:
        P = -P
    return P


def reproject_linear(P, xyz):
    proj = np.c_[xyz, np.ones(len(xyz))] @ P.T
    return proj[:, :2] / proj[:, 2:3]


def ransac_dlt(xyz, uv, thresh_px=8.0, max_iters=2000, conf=0.999, seed=0):
    '''RANSAC over minimal 8-point DLT samples -> (P_refit, inlier_mask, info).

    Needed because LED reflections can contaminate ~half the correspondences,
    which corrupts a plain all-points DLT beyond what soft_l1 + MAD rejection
    can recover from.
    '''
    n = len(xyz)
    sample_size = 8
    rng = np.random.default_rng(seed)
    best_count, best_inliers = -1, None
    n_needed = max_iters
    it = 0
    while it < min(n_needed, max_iters):
        it += 1
        idx = rng.choice(n, sample_size, replace=False)
        sv = np.linalg.svd(xyz[idx] - xyz[idx].mean(axis=0), compute_uv=False)
        if sv[2] / sv[0] < 0.01:  # near-planar sample -> degenerate DLT
            continue
        try:
            P = fit_dlt(xyz[idx], uv[idx])
            errs = np.linalg.norm(reproject_linear(P, xyz) - uv, axis=1)
        except np.linalg.LinAlgError:
            continue
        inliers = errs < thresh_px
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
            w = max(count / n, 1e-6)
            n_needed = np.log(1 - conf) / np.log1p(-min(w ** sample_size,
                                                        1 - 1e-12))
    if best_count < 12:
        raise ValueError('RANSAC found only %d consensus points (need 12) -- '
                         'correspondences may be misaligned; check the merge'
                         % best_count)
    # refit on the consensus and re-threshold once (lets points re-enter)
    P = fit_dlt(xyz[best_inliers], uv[best_inliers])
    errs = np.linalg.norm(reproject_linear(P, xyz) - uv, axis=1)
    inliers = errs < thresh_px
    if inliers.sum() >= 12:
        P = fit_dlt(xyz[inliers], uv[inliers])
    else:
        inliers = best_inliers
    info = {'n_iters': int(it), 'thresh_px': float(thresh_px),
            'n_consensus': int(inliers.sum()),
            'consensus_fraction': float(inliers.sum() / n)}
    return P, inliers, info


def decompose_P(P):
    '''P -> (K, R, rvec, tvec) with positive K diagonal and det(R)=+1.'''
    K, R, C_hom = cv2.decomposeProjectionMatrix(P)[:3]
    # enforce positive diagonal of K
    S = np.diag(np.sign(np.diag(K)))
    K = K @ S
    R = S @ R
    if np.linalg.det(R) < 0:
        R = -R
    K = K / K[2, 2]
    C = (C_hom[:3] / C_hom[3]).ravel()
    tvec = -R @ C
    rvec = cv2.Rodrigues(R)[0].ravel()
    return K, R, rvec, tvec


# ------------------------------------------------------------ nonlinear refine

def _pack(K, rvec, tvec, dist, opts):
    p = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist[0], dist[1]]
    if opts['tangential']:
        p += [dist[2], dist[3]]
    if opts['k3_free']:
        p += [dist[4]]
    return np.array(p + list(rvec) + list(tvec))


def _unpack(p, opts):
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
    dist = np.zeros(5)
    dist[0], dist[1] = p[4], p[5]
    i = 6
    if opts['tangential']:
        dist[2], dist[3] = p[i], p[i + 1]
        i += 2
    if opts['k3_free']:
        dist[4] = p[i]
        i += 1
    rvec = p[i:i + 3]
    tvec = p[i + 3:i + 6]
    return K, dist, rvec, tvec


def _bounds(f0, image_size, opts):
    w, h = image_size
    lo = [0.2 * f0, 0.2 * f0, 0.0, 0.0, -1.0, -1.0]
    hi = [5.0 * f0, 5.0 * f0, float(w), float(h), 1.0, 1.0]
    if opts['tangential']:
        lo += [-0.05, -0.05]
        hi += [0.05, 0.05]
    if opts['k3_free']:
        lo += [-2.0]
        hi += [2.0]
    lo += [-np.inf] * 6
    hi += [np.inf] * 6
    return np.array(lo), np.array(hi)


def _project(xyz, K, dist, rvec, tvec):
    proj, _ = cv2.projectPoints(xyz.reshape(-1, 1, 3), rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def _residuals(p, xyz, uv, opts):
    K, dist, rvec, tvec = _unpack(p, opts)
    return (_project(xyz, K, dist, rvec, tvec) - uv).ravel()


def refine(xyz, uv, K0, rvec0, tvec0, image_size, opts):
    '''Robust nonlinear refinement of the full camera model.

    scipy least_squares with soft_l1 loss (down-weights outliers) and bounds
    (keeps the distortion polynomial sane in uncovered sensor regions) --
    cv2.calibrateCamera offers neither for a single non-planar view.
    Note: braid covariance weighting (1/sqrt(P00+P11+P22)) could be added to
    _residuals if data-grade accuracy is ever needed.
    '''
    p0 = _pack(K0, rvec0, tvec0, np.zeros(5), opts)
    lo, hi = _bounds(0.5 * (K0[0, 0] + K0[1, 1]), image_size, opts)
    p0 = np.clip(p0, lo, hi)
    res = scipy.optimize.least_squares(_residuals, p0, args=(xyz, uv, opts),
                                       bounds=(lo, hi), method='trf',
                                       loss='soft_l1', f_scale=2.0,
                                       x_scale='jac')
    n_finite = np.isfinite(lo).sum()
    on_bound = ((np.abs(res.x - lo) < BOUND_HIT_TOL) |
                (np.abs(res.x - hi) < BOUND_HIT_TOL))[:n_finite]
    if on_bound.any():
        print('WARNING: %d fit parameter(s) sit on their bound -- the fit may '
              'be ill-posed; inspect the diagnostics' % on_bound.sum())
    K, dist, rvec, tvec = _unpack(res.x, opts)
    errs = np.linalg.norm(_project(xyz, K, dist, rvec, tvec) - uv, axis=1)
    return {'K': K, 'dist': dist, 'rvec': rvec, 'tvec': tvec}, errs


def error_stats(errs):
    return {'median_px': float(np.median(errs)),
            'rms_px': float(np.sqrt(np.mean(errs ** 2))),
            'p95_px': float(np.percentile(errs, 95))}


def robust_fit(xyz, uv, image_size, opts, outlier_px=None, n_iters=2,
               ransac_px=8.0, use_ransac=True):
    '''RANSAC DLT init -> refine -> outlier rejection -> refine (x n_iters).'''
    ransac_info = None
    if use_ransac:
        P_dlt, inliers, ransac_info = ransac_dlt(xyz, uv, thresh_px=ransac_px)
        print('RANSAC: %d/%d consensus (%.0f%%) at %.1f px in %d iterations'
              % (ransac_info['n_consensus'], len(xyz),
                 100 * ransac_info['consensus_fraction'], ransac_px,
                 ransac_info['n_iters']))
    else:
        P_dlt = fit_dlt(xyz, uv)
        inliers = np.ones(len(xyz), dtype=bool)
    dlt_errs = np.linalg.norm(reproject_linear(P_dlt, xyz) - uv, axis=1)
    K0, _R0, rvec0, tvec0 = decompose_P(P_dlt)

    calib, _ = refine(xyz[inliers], uv[inliers], K0, rvec0, tvec0,
                      image_size, opts)
    errs = np.linalg.norm(_project(xyz, calib['K'], calib['dist'],
                                   calib['rvec'], calib['tvec']) - uv, axis=1)
    threshold = None
    for _ in range(n_iters):
        if outlier_px is not None:
            threshold = float(outlier_px)
        else:
            mad = np.median(np.abs(errs[inliers] - np.median(errs[inliers])))
            threshold = max(3 * 1.4826 * mad + np.median(errs[inliers]), 2.0)
        new_inliers = errs <= threshold
        if new_inliers.sum() < 12:
            print('WARNING: outlier rejection left <12 points; keeping all')
            break
        inliers = new_inliers
        calib, errs = refine(xyz[inliers], uv[inliers],
                             calib['K'], calib['rvec'], calib['tvec'],
                             image_size, opts)
        # recompute errors on ALL points for the next rejection round
        full_errs = np.linalg.norm(_project(xyz, calib['K'], calib['dist'],
                                            calib['rvec'], calib['tvec']) - uv,
                                   axis=1)
        errs = full_errs

    if calib['K'][0, 0] > 0 and abs(calib['K'][0, 0] - calib['K'][1, 1]) \
            / calib['K'][0, 0] > 0.05:
        print('WARNING: fx (%.1f) and fy (%.1f) differ by >5%% -- single-view '
              'fit may be poorly constrained' % (calib['K'][0, 0], calib['K'][1, 1]))
    cx, cy = calib['K'][0, 2], calib['K'][1, 2]
    w, h = image_size
    if abs(cx - w / 2) > 0.15 * w or abs(cy - h / 2) > 0.15 * h:
        print('WARNING: principal point (%.0f, %.0f) is far from the image '
              'center -- expected for single-view fits, fine for visualization'
              % (cx, cy))

    stats = {'n_pairs_total': int(len(xyz)),
             'n_inliers': int(inliers.sum()),
             'n_outliers': int((~inliers).sum()),
             'outlier_threshold_px': float(threshold) if threshold else None,
             'ransac': ransac_info,
             'stages': {'linear_dlt': error_stats(dlt_errs[inliers]),
                        'full_model': error_stats(errs[inliers])}}
    return calib, P_dlt, inliers, errs, stats


# ---------------------------------------------------------------------- output

def build_yaml_dict(calib, stats, geometry, metadata, opts, dataset_dir):
    K, dist, rvec, tvec = calib['K'], calib['dist'], calib['rvec'], calib['tvec']
    R = cv2.Rodrigues(np.asarray(rvec))[0]
    P_linear = K @ np.c_[R, np.asarray(tvec).reshape(3, 1)]
    return {
        'schema': 'braid_tools/aux_camera_calibration-v1',
        'camera_name': metadata.get('camera_name', ''),
        'image_width': metadata.get('image_width'),
        'image_height': metadata.get('image_height'),
        'model': 'plumb_bob',
        'K': np.asarray(K).tolist(),
        'distortion': np.asarray(dist).tolist(),
        'rvec': np.asarray(rvec).ravel().tolist(),
        'tvec': np.asarray(tvec).ravel().tolist(),
        'R': R.tolist(),
        'P_linear': P_linear.tolist(),
        'fit': {**{k: v for k, v in stats.items() if k != 'stages'},
                'options': {'k3_free': opts['k3_free'],
                            'tangential': opts['tangential'],
                            'loss': 'soft_l1', 'f_scale_px': 2.0},
                'stages': stats['stages']},
        'geometry': {**geometry,
                     'note': 'distortion model only validated inside '
                             'valid_pixel_region; extrapolation toward '
                             'uncovered sensor edges is unconstrained'},
        'provenance': {'dataset_dir': os.path.basename(os.path.abspath(dataset_dir)),
                       'created_iso': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                       'script': os.path.basename(__file__),
                       'cv2_version': cv2.__version__},
    }


def load_camera_calibration(yaml_path):
    '''Load a camera_calibration.yaml into a dict of numpy arrays.

    (Named distinctly from braid_analysis.braid_2d_analysis.load_calibration,
    which loads braidz multi-camera calibrations.)
    '''
    d = yaml.safe_load(open(yaml_path))
    return {'camera_name': d.get('camera_name', ''),
            'image_width': d.get('image_width'),
            'image_height': d.get('image_height'),
            'K': np.array(d['K']),
            'dist': np.array(d['distortion']),
            'rvec': np.array(d['rvec']),
            'tvec': np.array(d['tvec']),
            'R': np.array(d['R']),
            'P_linear': np.array(d['P_linear'])}


def project_points(xyz, calib):
    '''Project world points through the full model. Returns (u, v, in_front).

    Points behind the camera are NaN'd (cv2.projectPoints silently mirrors
    them into the image otherwise).
    '''
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    uv = _project(xyz, calib['K'], calib['dist'], calib['rvec'], calib['tvec'])
    z_cam = (calib['R'] @ xyz.T).T[:, 2] + calib['tvec'][2]
    in_front = z_cam > 0
    uv[~in_front] = np.nan
    return uv[:, 0], uv[:, 1], in_front


# ----------------------------------------------------------------- diagnostics

def make_diagnostics(dataset_dir, xyz, uv, calib, P_dlt, inliers, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    uv_dlt = reproject_linear(P_dlt, xyz)
    u_full, v_full, _ = project_points(xyz, {**calib,
                                             'R': cv2.Rodrigues(calib['rvec'])[0]})
    uv_full = np.c_[u_full, v_full]
    r_dlt = uv_dlt - uv
    r_full = uv_full - uv
    e_dlt = np.linalg.norm(r_dlt, axis=1)
    e_full = np.linalg.norm(r_full, axis=1)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4])

    for col, (r, e, title) in enumerate([(r_dlt, e_dlt, 'linear DLT'),
                                         (r_full, e_full, 'full model')]):
        ax = fig.add_subplot(gs[0, col])
        q = ax.quiver(uv[inliers, 0], uv[inliers, 1],
                      r[inliers, 0], r[inliers, 1], e[inliers],
                      angles='xy', scale_units='xy', scale=0.05, width=0.003,
                      cmap='viridis')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title('%s residuals (median %.2f px)  [arrows x20]'
                     % (title, np.median(e[inliers])))
        fig.colorbar(q, ax=ax, shrink=0.8, label='px')

    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, np.percentile(e_dlt[inliers], 99.5), 60)
    ax.hist(e_dlt[inliers], bins=bins, alpha=0.6, label='linear DLT')
    ax.hist(e_full[inliers], bins=bins, alpha=0.6, label='full model')
    ax.set_xlabel('reprojection error (px)')
    ax.set_ylabel('count')
    ax.legend()
    ax.set_title('inlier residuals (%d inliers, %d outliers, %.0f%% kept)'
                 % (inliers.sum(), (~inliers).sum(),
                    100.0 * inliers.sum() / len(inliers)))

    ax = fig.add_subplot(gs[1, :])
    ref = os.path.join(dataset_dir, 'reference.png')
    img = cv2.imread(ref) if os.path.exists(ref) else None
    if img is not None:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.set_facecolor('0.1')
        ax.set_xlim(uv[:, 0].min() - 50, uv[:, 0].max() + 50)
        ax.set_ylim(uv[:, 1].max() + 50, uv[:, 1].min() - 50)
    ax.plot(uv[inliers, 0], uv[inliers, 1], '.', color='cyan', ms=2,
            label='2D detections (%d)' % inliers.sum())
    ax.plot(uv_full[inliers, 0], uv_full[inliers, 1], '.', color='red', ms=1,
            alpha=0.7, label='reprojected braid 3D')
    if (~inliers).any():
        ax.plot(uv[~inliers, 0], uv[~inliers, 1], 'x', color='orange', ms=3,
                alpha=0.4, label='rejected outliers (%d)' % (~inliers).sum())
    ax.legend(loc='upper right')
    ax.set_title('reference image with reprojected trajectory')

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------- selftest

def make_synthetic_dataset(seed, tmpdir, n=4000, noise_px=0.5, outlier_frac=0.01,
                           reflection_frac=0.0):
    rng = np.random.default_rng(seed)
    K = np.array([[1800.0, 0, 940.0], [0, 1800.0, 610.0], [0, 0, 1.0]])
    dist = np.array([-0.25, 0.08, 1e-3, -5e-4, 0.0])
    rvec = np.array([0.1, -0.15, 0.05]) + rng.normal(0, 0.02, 3)
    tvec = np.array([0.05, -0.1, 1.8]) + rng.normal(0, 0.02, 3)

    t = np.linspace(0, 60, n)
    xyz = np.zeros((n, 3))
    for ax in range(3):
        for _ in range(4):
            f = rng.uniform(0.05, 0.5)
            xyz[:, ax] += rng.uniform(0.05, 0.35) * np.sin(2 * np.pi * f * t
                                                           + rng.uniform(0, 6.28))
    uv = _project(xyz, K, dist, rvec, tvec) + rng.normal(0, noise_px, (n, 2))

    n_out = int(outlier_frac * n)
    out_idx = rng.choice(n, n_out, replace=False)
    uv[out_idx] += rng.uniform(100, 300, (n_out, 2)) * rng.choice([-1, 1], (n_out, 2))

    # reflection-like contamination: contiguous runs replaced by a mirrored,
    # offset copy of the true trajectory (spatially coherent, like a real
    # LED reflection the point detector locks onto)
    n_refl = int(reflection_frac * n)
    refl_idx = np.array([], dtype=int)
    if n_refl:
        starts = rng.choice(n - 200, max(1, n_refl // 200), replace=False)
        refl_idx = np.unique(np.concatenate(
            [np.arange(s, s + 200) for s in starts]))[:n_refl]
        mirror = np.array([1920.0, 0.0]) - uv[refl_idx] * np.array([1.0, -1.0])
        uv[refl_idx] = mirror + rng.uniform(-150, 150, 2)
    out_idx = np.union1d(out_idx, refl_idx)
    n_out = len(out_idx)

    keep = ((uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1200))
    df = pd.DataFrame({'x': xyz[keep, 0], 'y': xyz[keep, 1], 'z': xyz[keep, 2],
                       'x_px': uv[keep, 0], 'y_px': uv[keep, 1]})
    df.to_csv(os.path.join(tmpdir, 'merged.csv'), index=False)
    with open(os.path.join(tmpdir, 'metadata.yaml'), 'w') as f:
        yaml.safe_dump({'camera_name': 'synthetic', 'image_width': 1920,
                        'image_height': 1200}, f)
    n_out_kept = int(np.isin(np.where(keep)[0], out_idx).sum())
    truth = {'K': K, 'dist': dist, 'rvec': rvec, 'tvec': tvec,
             'n_outliers': n_out_kept, 'noise_px': noise_px}
    return truth


def selftest():
    import tempfile
    ok = True
    cases = [(1, 0.0), (2, 0.0), (3, 0.0), (1, 0.35), (2, 0.4)]
    for seed, reflection_frac in cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            truth = make_synthetic_dataset(seed, tmpdir,
                                           reflection_frac=reflection_frac)
            report = run(tmpdir, argparse.Namespace(
                no_tangential=False, k3=False, outlier_px=None,
                ransac_px=8.0, no_ransac=False,
                cross_check=False, no_plot=True))
            calib = load_camera_calibration(os.path.join(tmpdir,
                                                         'camera_calibration.yaml'))
            fx_err = abs(calib['K'][0, 0] - truth['K'][0, 0]) / truth['K'][0, 0]
            fy_err = abs(calib['K'][1, 1] - truth['K'][1, 1]) / truth['K'][1, 1]
            c_err = np.hypot(calib['K'][0, 2] - truth['K'][0, 2],
                             calib['K'][1, 2] - truth['K'][1, 2])
            k_err = np.abs(calib['dist'][:2] - truth['dist'][:2]).max()
            med = report['fit']['stages']['full_model']['median_px']
            n_out = report['fit']['n_outliers']
            passed = (fx_err < 0.015 and fy_err < 0.015 and c_err < 10
                      and k_err < 0.03 and med < 2 * truth['noise_px']
                      and 0.5 * truth['n_outliers'] <= n_out <= 1.5 * truth['n_outliers']
                      and med < report['fit']['stages']['linear_dlt']['median_px'])
            ok &= passed
            print('seed %d (refl %.0f%%): fx err %.2f%%, c err %.1f px, '
                  'k err %.3f, median %.2f px, outliers %d (true %d) -> %s'
                  % (seed, 100 * reflection_frac, 100 * fx_err, c_err, k_err,
                     med, n_out, truth['n_outliers'],
                     'PASS' if passed else 'FAIL'))
    print('selftest:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


# ------------------------------------------------------------------------ main

def run(dataset_dir, args):
    df, xyz, uv, metadata = load_merged(dataset_dir)
    image_size = (metadata.get('image_width') or 1920,
                  metadata.get('image_height') or 1200)
    geometry = check_geometry(xyz, uv, image_size)
    opts = {'tangential': not args.no_tangential, 'k3_free': args.k3}

    calib, P_dlt, inliers, errs, stats = robust_fit(
        xyz, uv, image_size, opts, outlier_px=args.outlier_px,
        ransac_px=args.ransac_px, use_ransac=not args.no_ransac)

    # reflections must not extend the "validated" distortion region
    uv_in = uv[inliers]
    region = {'x_min': float(uv_in[:, 0].min()), 'x_max': float(uv_in[:, 0].max()),
              'y_min': float(uv_in[:, 1].min()), 'y_max': float(uv_in[:, 1].max())}
    w, h = image_size
    geometry['valid_pixel_region'] = region
    geometry['pixel_coverage_fraction'] = float(
        (region['x_max'] - region['x_min']) *
        (region['y_max'] - region['y_min']) / float(w * h))

    if args.cross_check:
        _cross_check(xyz[inliers], uv[inliers], calib, image_size, opts)

    out = build_yaml_dict(calib, stats, geometry, metadata, opts, dataset_dir)
    yaml_path = os.path.join(dataset_dir, 'camera_calibration.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)

    if not args.no_plot:
        make_diagnostics(dataset_dir, xyz, uv, calib, P_dlt, inliers,
                         os.path.join(dataset_dir, 'calibration_diagnostics.png'))

    print('\n%d/%d inliers (threshold %.2f px)'
          % (stats['n_inliers'], stats['n_pairs_total'],
             stats['outlier_threshold_px']))
    for stage, s in stats['stages'].items():
        print('  %-12s median %.2f px   rms %.2f px   p95 %.2f px'
              % (stage, s['median_px'], s['rms_px'], s['p95_px']))
    K, dist = calib['K'], calib['dist']
    print('  fx %.1f  fy %.1f  cx %.1f  cy %.1f' % (K[0, 0], K[1, 1],
                                                    K[0, 2], K[1, 2]))
    print('  dist [k1 k2 p1 p2 k3] = [%s]' % ' '.join('%.4g' % d for d in dist))
    print('wrote %s' % yaml_path)
    return out


def _cross_check(xyz, uv, calib, image_size, opts):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if not opts['k3_free']:
        flags |= cv2.CALIB_FIX_K3
    if not opts['tangential']:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    ret, K2, dist2, _rv, _tv = cv2.calibrateCamera(
        [xyz.astype(np.float32).reshape(-1, 1, 3)],
        [uv.astype(np.float32).reshape(-1, 1, 2)],
        image_size, calib['K'].copy(), calib['dist'].copy().reshape(1, -1),
        flags=flags)
    print('cross-check vs cv2.calibrateCamera (rms %.2f px):' % ret)
    print('  d_fx %.1f  d_fy %.1f  d_cx %.1f  d_cy %.1f'
          % (K2[0, 0] - calib['K'][0, 0], K2[1, 1] - calib['K'][1, 1],
             K2[0, 2] - calib['K'][0, 2], K2[1, 2] - calib['K'][1, 2]))
    print('  d_dist %s' % np.round(np.ravel(dist2)[:5] - calib['dist'], 4))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_dir', nargs='?',
                        help='dataset directory containing merged.csv')
    parser.add_argument('--no-tangential', action='store_true',
                        help='fix tangential distortion p1=p2=0')
    parser.add_argument('--k3', action='store_true',
                        help='also fit the k3 (r^6) radial term')
    parser.add_argument('--outlier-px', type=float, default=None,
                        help='fixed outlier threshold (default: adaptive 3*MAD)')
    parser.add_argument('--ransac-px', type=float, default=8.0,
                        help='RANSAC inlier threshold for the DLT init')
    parser.add_argument('--no-ransac', action='store_true',
                        help='plain all-points DLT init (pre-RANSAC behavior; '
                             'fails on reflection-heavy datasets)')
    parser.add_argument('--cross-check', action='store_true',
                        help='also fit with cv2.calibrateCamera and print deltas')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.dataset_dir:
        parser.error('dataset_dir required unless --selftest')
    run(os.path.expanduser(args.dataset_dir), args)


if __name__ == '__main__':
    main()
