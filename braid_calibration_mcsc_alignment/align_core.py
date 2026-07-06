#!/usr/bin/env python3
"""Core machinery for aligning an MCSC multi-camera calibration to arena geometry.

GUI-free and importable. Automates "step 5" of the braid calibration workflow:
given an unaligned calibration XML (<multi_camera_reconstructor>), a .braidz
dataset in which an LED wand traced the arena surfaces (plus interior axis
arrows), and a flydra-style stimxml arena box, find the similarity transform
M = [[s*R, t], [0, 1]] (reflections allowed) such that transformed points
X' = s*R*X + t lie on the arena box surface. Cameras transform as
P' = P @ inv(M), leaving reprojection of transformed points unchanged.

Run ``python3 align_core.py --selftest`` to exercise everything synthetically.
"""

import argparse
import gzip
import io
import itertools
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

NON_LINEAR_KEYS = ["fc1", "fc2", "cc1", "cc2", "k1", "k2", "p1", "p2", "alpha_c"]


def load_wand_points(braidz_path, min_points=100):
    """Load all 3D tracked points (all obj_ids) from a .braidz as (N, 3)."""
    try:
        from braid_analysis import braid_filemanager

        df = braid_filemanager.load_filename_as_dataframe_3d(braidz_path)
    except ImportError:
        with zipfile.ZipFile(braidz_path) as z:
            with z.open("kalman_estimates.csv.gz") as f:
                df = pd.read_csv(
                    io.BytesIO(gzip.decompress(f.read())), comment="#"
                )
    X = df[["x", "y", "z"]].to_numpy(float)
    good = np.isfinite(X).all(axis=1)
    n_bad = int((~good).sum())
    if n_bad:
        print(f"warning: dropped {n_bad} rows with NaN/inf coordinates")
    X = X[good]
    if len(X) < min_points:
        raise ValueError(
            f"only {len(X)} valid 3D points in {braidz_path}; need >= {min_points}"
        )
    return X


def load_stim_box(stimxml_path):
    """Parse a flydra stimxml cubic_arena. Returns dict with lo, hi, verts."""
    root = ET.parse(stimxml_path).getroot()
    verts = []
    for vert in root.findall(".//cubic_arena/verts4x4/vert"):
        verts.append([float(v) for v in vert.text.split(",")])
    verts = np.array(verts, dtype=float)
    if verts.shape != (8, 3):
        raise ValueError(
            f"expected 8 <vert> entries in {stimxml_path}, got {verts.shape}"
        )
    return {
        "verts": verts,
        "lo": verts.min(axis=0),
        "hi": verts.max(axis=0),
    }


def load_calibration_xml(xml_path):
    """Parse a <multi_camera_reconstructor> XML into a list of camera dicts."""
    root = ET.parse(xml_path).getroot()
    if root.tag != "multi_camera_reconstructor":
        raise ValueError(f"unexpected root element <{root.tag}> in {xml_path}")
    cams = []
    for scc in root.findall("single_camera_calibration"):
        rows = scc.find("calibration_matrix").text.split(";")
        P = np.array([[float(v) for v in row.split()] for row in rows])
        if P.shape != (3, 4):
            raise ValueError(f"bad calibration_matrix shape {P.shape}")
        res = tuple(int(v) for v in scc.find("resolution").text.split())
        nlp = {}
        nlp_el = scc.find("non_linear_parameters")
        if nlp_el is not None:
            for child in nlp_el:
                nlp[child.tag] = child.text  # keep original text verbatim
        cams.append(
            {
                "cam_id": scc.find("cam_id").text,
                "P": P,
                "resolution": res,
                "non_linear_parameters": nlp,
            }
        )
    if not cams:
        raise ValueError(f"no <single_camera_calibration> entries in {xml_path}")
    return cams


def _fmt(v):
    # '%.17g'-style: full float round-trip precision, '1' for 1.0 (matches input)
    return format(v, ".17g")


def write_calibration_xml(cams, out_path):
    """Write cameras (with final P) in the mcsc XML schema, matching input style."""
    lines = ["<multi_camera_reconstructor>"]
    for cam in cams:
        mat = "; ".join(" ".join(_fmt(v) for v in row) for row in cam["P"])
        lines.append("    <single_camera_calibration>")
        lines.append(f"      <cam_id>{cam['cam_id']}</cam_id>")
        lines.append(f"      <calibration_matrix>{mat}</calibration_matrix>")
        w, h = cam["resolution"]
        lines.append(f"      <resolution>{w} {h}</resolution>")
        if cam["non_linear_parameters"]:
            lines.append("      <non_linear_parameters>")
            for key, text in cam["non_linear_parameters"].items():
                lines.append(f"        <{key}>{text}</{key}>")
            lines.append("      </non_linear_parameters>")
        lines.append("    </single_camera_calibration>")
    lines.append("</multi_camera_reconstructor>")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Similarity-transform algebra
# ---------------------------------------------------------------------------


def make_M(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * np.asarray(R)
    M[:3, 3] = np.asarray(t)
    return M


def decompose_M(M):
    """Split M into scale, det sign, orthonormal Q (det +/-1), translation."""
    A = M[:3, :3]
    s = abs(np.linalg.det(A)) ** (1.0 / 3.0)
    Q = A / s
    return {
        "s": s,
        "det_sign": 1 if np.linalg.det(A) > 0 else -1,
        "Q": Q,
        "t": M[:3, 3].copy(),
    }


def invert_similarity(M):
    A = M[:3, :3]
    t = M[:3, 3]
    d = decompose_M(M)
    Ainv = A.T / (d["s"] ** 2)  # (s*Q)^-1 = Q^T / s, Q orthonormal (det +/-1)
    Minv = np.eye(4)
    Minv[:3, :3] = Ainv
    Minv[:3, 3] = -Ainv @ t
    return Minv


def transform_points(M, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return X @ M[:3, :3].T + M[:3, 3]


def transform_P(P, M):
    """P' = P @ inv(M), renormalized so P'[2,3] = 1 (input file convention)."""
    Pp = P @ invert_similarity(M)
    if abs(Pp[2, 3]) < 1e-9:
        raise ValueError(
            "transformed P has P[2,3] ~= 0 (world origin on the camera's "
            "principal plane); cannot renormalize"
        )
    return Pp / Pp[2, 3]


def reproject(P, xyz):
    """Project world points with 3x4 P (braid_2d_analysis.reproject convention)."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    proj = P @ np.hstack([xyz, np.ones((len(xyz), 1))]).T
    return proj[0] / proj[2], proj[1] / proj[2]


# ---------------------------------------------------------------------------
# Box-surface distance and robust fit
# ---------------------------------------------------------------------------


def box_surface_distance(points, lo, hi):
    """Unsigned distance from each point to the surface of an AABB, (N,3)->(N,)."""
    c = (np.asarray(lo) + np.asarray(hi)) / 2.0
    h = (np.asarray(hi) - np.asarray(lo)) / 2.0
    q = np.abs(np.atleast_2d(points) - c) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return np.abs(outside + inside)


def box_symmetry_group(extents, rtol=1e-6):
    """Signed permutation matrices mapping a box with these extents onto itself.

    Includes improper (reflecting) elements — MCSC calibrations can be
    mirror-imaged, so reflected starts are mandatory.
    """
    extents = np.asarray(extents, dtype=float)
    mats = []
    for perm in itertools.permutations(range(3)):
        if not np.allclose(extents[list(perm)], extents, rtol=rtol):
            continue
        for signs in itertools.product([1.0, -1.0], repeat=3):
            F = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                F[i, p] = s
            mats.append(F)
    return mats


def _apply_theta(theta, F, X):
    s = np.exp(theta[0])
    R = Rotation.from_rotvec(theta[1:4]).as_matrix()
    return X @ (s * R @ F).T + theta[4:7]


def _theta_to_M(theta, F):
    return make_M(np.exp(theta[0]), Rotation.from_rotvec(theta[1:4]).as_matrix() @ F, theta[4:7])


def initial_guess(X, lo, hi):
    """Robust PCA-based init. Returns (s0, R0, c_data) with R0 mapping data->box axes."""
    c_d = np.median(X, axis=0)
    Xc = X - c_d
    evals, evecs = np.linalg.eigh(np.cov(Xc.T))
    U = evecs[:, ::-1]  # columns = data axes, descending variance
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    proj = Xc @ U
    # percentile extents so arrow tips / stray tracks don't inflate the span
    e = np.percentile(proj, 97.5, axis=0) - np.percentile(proj, 2.5, axis=0)
    b = np.asarray(hi) - np.asarray(lo)
    order = np.argsort(b)[::-1]  # box axes sorted by descending extent
    s0 = float(np.median(b[order] / e))
    Rmap = np.zeros((3, 3))
    for k in range(3):
        Rmap[order[k], k] = 1.0
    R0 = Rmap @ U.T
    if np.linalg.det(R0) < 0:
        R0 = Rmap @ np.diag([1.0, 1.0, -1.0]) @ U.T
    return s0, R0, c_d


def compute_metrics(X, lo, hi, M, trim=0.03):
    d = box_surface_distance(transform_points(M, X), lo, hi)
    trimmed = d[d < trim]
    return {
        "n_points": int(len(d)),
        "median_abs_dist_m": float(np.median(d)),
        "p90_abs_dist_m": float(np.percentile(d, 90)),
        "inlier_frac_1cm": float(np.mean(d < 0.01)),
        "inlier_frac_2cm": float(np.mean(d < 0.02)),
        "trimmed_median_m": float(np.median(trimmed)) if len(trimmed) else np.nan,
        "trimmed_frac": float(len(trimmed) / len(d)),
    }


def fit_similarity(X, lo, hi, theta0, F):
    """Three-round trimmed robust fit with fixed reflection F. None if collapsed."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    c_box = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    theta0 = np.asarray(theta0, dtype=float)

    lb = np.full(7, -np.inf)
    ub = np.full(7, np.inf)
    lb[0], ub[0] = theta0[0] - np.log(2), theta0[0] + np.log(2)
    lb[4:7], ub[4:7] = c_box - 2 * diag, c_box + 2 * diag
    theta0 = np.clip(theta0, lb, ub)
    x_scale = [0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]

    def solve(theta_init, Xfit, loss, f_scale):
        return least_squares(
            lambda th: box_surface_distance(_apply_theta(th, F, Xfit), lo, hi),
            theta_init, bounds=(lb, ub), method="trf",
            loss=loss, f_scale=f_scale, x_scale=x_scale,
        ).x

    theta = solve(theta0, X, "soft_l1", 0.02)
    for thresh_floor, loss, f_scale in [(0.05, "soft_l1", 0.01), (0.03, "linear", 0.01)]:
        r = box_surface_distance(_apply_theta(theta, F, X), lo, hi)
        sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = r < max(thresh_floor, 3 * sigma)
        if keep.sum() < 50:
            return None
        theta = solve(theta, X[keep], loss, f_scale)

    # anti-collapse sanity checks
    if theta[0] <= lb[0] + 1e-6 or theta[0] >= ub[0] - 1e-6:
        return None
    Xt = _apply_theta(theta, F, X)
    b = hi - lo
    ax_long = int(np.argmax(b))
    span = np.percentile(Xt[:, ax_long], 97.5) - np.percentile(Xt[:, ax_long], 2.5)
    if span < 0.5 * b[ax_long]:
        return None

    M = _theta_to_M(theta, F)
    return {"theta": theta, "F": F, "M": M}


def autofit(X, lo, hi, max_fit_points=8000, seed=0):
    """Multi-start robust fit over the box symmetry group. Returns best result.

    For a box with equal extents on two axes all symmetry-related solutions
    score identically — the winner is an arbitrary branch; the GUI flip/90deg
    controls hop between branches.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    c_box = (lo + hi) / 2.0
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), max_fit_points, replace=False)] if len(X) > max_fit_points else X

    s0, R0, c_d = initial_guess(Xs, lo, hi)
    F_reflect = np.diag([-1.0, 1.0, 1.0])
    results = []
    for S in box_symmetry_group(hi - lo):
        Rs = S @ R0  # symmetry hop applied in box frame
        if np.linalg.det(Rs) < 0:
            F, Rth = F_reflect, Rs @ F_reflect
        else:
            F, Rth = np.eye(3), Rs
        rotvec = Rotation.from_matrix(Rth).as_rotvec()
        t0 = c_box - s0 * Rs @ c_d
        theta0 = np.concatenate([[np.log(s0)], rotvec, t0])
        res = fit_similarity(Xs, lo, hi, theta0, F)
        if res is not None:
            res["score"] = float(np.median(
                box_surface_distance(transform_points(res["M"], Xs), lo, hi)))
            results.append(res)
    if not results:
        raise RuntimeError(
            "all autofit starts collapsed or were rejected — is the dataset "
            "actually tracing the arena surfaces?"
        )
    best = min(results, key=lambda r: r["score"])
    best["metrics"] = compute_metrics(X, lo, hi, best["M"])
    return best


def refit_from_M(X, lo, hi, M_seed, max_fit_points=8000, seed=0):
    """Single-branch refit seeded from an existing transform (GUI 're-run autofit')."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), max_fit_points, replace=False)] if len(X) > max_fit_points else X
    d = decompose_M(M_seed)
    if d["det_sign"] < 0:
        F = np.diag([-1.0, 1.0, 1.0])
        Rth = d["Q"] @ F  # det +1, valid for rotvec
    else:
        F, Rth = np.eye(3), d["Q"]
    theta0 = np.concatenate(
        [[np.log(d["s"])], Rotation.from_matrix(Rth).as_rotvec(), d["t"]]
    )
    res = fit_similarity(Xs, lo, hi, theta0, F)
    if res is None:
        raise RuntimeError("seeded refit collapsed; adjust manually and retry")
    res["metrics"] = compute_metrics(X, lo, hi, res["M"])
    return res


# ---------------------------------------------------------------------------
# GUI transform composition
# ---------------------------------------------------------------------------


def compose_manual(M_fit, c_box, scale=1.0, rot_deg=(0, 0, 0), trans=(0, 0, 0),
                   flips=(1, 1, 1), R_coarse=None):
    """M_total = T(c)*T(dt)*Rz*Ry*Rx*R_coarse*F*s*T(-c)*M_fit.

    Manual factors pivot about the arena box center c_box so flips/rotations
    turn the already-aligned cloud in place.
    """
    if R_coarse is None:
        R_coarse = np.eye(3)
    R_fine = Rotation.from_euler("XYZ", rot_deg, degrees=True).as_matrix()
    A = R_fine @ R_coarse @ np.diag(np.asarray(flips, dtype=float)) * scale
    M_manual = np.eye(4)
    M_manual[:3, :3] = A
    M_manual[:3, 3] = np.asarray(c_box) - A @ np.asarray(c_box) + np.asarray(trans)
    return M_manual @ M_fit


# ---------------------------------------------------------------------------
# Camera geometry (reflection-safe: no K/R/t decomposition)
# ---------------------------------------------------------------------------


def camera_center(P):
    """World-space camera center = null space of P (via SVD)."""
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1]
    if abs(C[3]) < 1e-12:
        raise ValueError("camera center at infinity")
    return C[:3] / C[3]


def frustum_points(P, C, resolution, length=0.15):
    """Four image-corner ray endpoints at `length` from center C, world frame.

    Sign-agnostic: works for P of either overall sign and for calibrations
    containing a reflection (where det-forcing decompositions point backwards).
    """
    W, H = resolution
    corners = np.empty((4, 3))
    for i, (u, v) in enumerate([(0, 0), (W, 0), (W, H), (0, H)]):
        d = np.linalg.solve(P[:, :3], [u, v, 1.0])
        w = P[2] @ np.append(C + d, 1.0)  # homogeneous depth of test point
        if w < 0:
            d = -d
        corners[i] = C + length * d / np.linalg.norm(d)
    return corners


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def check_similarity(M, tol=1e-9):
    A = M[:3, :3]
    s = decompose_M(M)["s"]
    err = np.linalg.norm(A.T @ A - s**2 * np.eye(3))
    if err > tol * max(1.0, s**2):
        raise AssertionError(f"M is not a similarity transform (|A^T A - s^2 I| = {err:g})")


def check_reprojection_invariance(cams, M, X, n=25, tol=1e-6, rng=None):
    """reproject(P, X) must equal reproject(P', M X) to within tol pixels."""
    rng = rng or np.random.default_rng(0)
    Xs = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Xt = transform_points(M, Xs)
    worst = 0.0
    for cam in cams:
        P, Pp = cam["P"], transform_P(cam["P"], M)
        w_old = (P @ np.hstack([Xs, np.ones((len(Xs), 1))]).T)[2]
        w_new = (Pp @ np.hstack([Xt, np.ones((len(Xt), 1))]).T)[2]
        ok = (np.abs(w_old) > 1e-9) & (np.abs(w_new) > 1e-9)
        if not ok.any():
            continue
        u0, v0 = reproject(P, Xs[ok])
        u1, v1 = reproject(Pp, Xt[ok])
        worst = max(worst, float(np.max(np.hypot(u1 - u0, v1 - v0))))
    if worst > tol:
        raise AssertionError(
            f"reprojection changed by up to {worst:g} px after transform (tol {tol:g})"
        )
    return worst


def check_roundtrip(cams_written, out_path):
    """Written XML must re-parse to the same matrices and identical passthrough text."""
    reread = load_calibration_xml(out_path)
    if len(reread) != len(cams_written):
        raise AssertionError("camera count changed in round trip")
    for a, b in zip(cams_written, reread):
        if a["cam_id"] != b["cam_id"] or a["resolution"] != b["resolution"]:
            raise AssertionError(f"cam_id/resolution mismatch for {a['cam_id']}")
        rel = np.max(np.abs(a["P"] - b["P"])) / max(1.0, np.max(np.abs(a["P"])))
        if rel > 1e-12:
            raise AssertionError(f"matrix round-trip error {rel:g} for {a['cam_id']}")
        if a["non_linear_parameters"] != b["non_linear_parameters"]:
            raise AssertionError(f"non_linear_parameters changed for {a['cam_id']}")


def run_save_checks(cams, M, X):
    """All pre-save numeric checks; raises AssertionError on failure."""
    check_similarity(M)
    worst = check_reprojection_invariance(cams, M, X)
    return worst


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------


def _synthetic_box_cloud(lo, hi, rng, n_surface=6000, noise=0.002):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ext = hi - lo
    # sample faces weighted by area
    areas = np.array([ext[1] * ext[2], ext[0] * ext[2], ext[0] * ext[1]]).repeat(2)
    face = rng.choice(6, size=n_surface, p=areas / areas.sum())
    X = lo + rng.random((n_surface, 3)) * ext
    axis = face // 2
    X[np.arange(n_surface), axis] = np.where(face % 2, hi[axis], lo[axis])
    # interior axis arrows from the box center
    c = (lo + hi) / 2.0
    arrows = []
    for ax in range(3):
        tt = rng.random(150)[:, None] * 0.4 * ext[ax]
        seg = np.tile(c, (150, 1))
        seg[:, ax] += tt[:, 0]
        arrows.append(seg)
    X = np.vstack([X] + arrows)
    return X + rng.normal(scale=noise, size=X.shape)


def _synthetic_cameras(lo, hi, rng, n_cams=6):
    c = (np.asarray(lo) + np.asarray(hi)) / 2.0
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 600], [0, 0, 1]])
    cams = []
    for i in range(n_cams):
        az = 2 * np.pi * i / n_cams
        C = c + 3.0 * np.array([np.cos(az), np.sin(az), 0.5])
        z = (c - C) / np.linalg.norm(c - C)  # optical axis toward box center
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        P = K @ np.hstack([R, (-R @ C)[:, None]])
        cams.append(
            {"cam_id": f"synth-{i}", "P": P / P[2, 3], "resolution": (1920, 1200),
             "non_linear_parameters": {k: "0" for k in NON_LINEAR_KEYS}}
        )
    return cams


def selftest():
    import tempfile
    import os

    rng = np.random.default_rng(42)
    lo = np.array([-0.9144, -0.30, 0.0])
    hi = np.array([0.9144, 0.30, 0.6])
    n_fail = 0

    def report(name, fn):
        nonlocal n_fail
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            n_fail += 1
            print(f"FAIL  {name}: {e}")

    X_box = _synthetic_box_cloud(lo, hi, rng)

    def fit_case(reflect):
        R_g = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        if reflect:
            R_g = R_g @ np.diag([-1.0, 1.0, 1.0])
        s_g = 1.8
        G = make_M(s_g, R_g, [0.4, -0.2, 1.0])
        X_data = transform_points(G, X_box)
        res = autofit(X_data, lo, hi, seed=1)
        m = res["metrics"]
        assert m["median_abs_dist_m"] < 0.02, f"median {m['median_abs_dist_m']:.4f}"
        assert m["trimmed_median_m"] < 0.004, f"trimmed median {m['trimmed_median_m']:.4f}"
        s_fit = decompose_M(res["M"])["s"]
        assert abs(s_fit * s_g - 1) < 0.01, f"scale product {s_fit * s_g:.4f}"
        # composed map must be a box symmetry: unit scale, tiny translation
        Cmp = res["M"] @ G
        dc = decompose_M(Cmp)
        assert abs(dc["s"] - 1) < 0.01
        c_box = (lo + hi) / 2.0
        drift = np.linalg.norm(transform_points(Cmp, c_box[None])[0] - c_box)
        assert drift < 0.01, f"box-center drift {drift:.4f} m"

    report("autofit recovers similarity (proper rotation)", lambda: fit_case(False))
    report("autofit recovers similarity (with reflection)", lambda: fit_case(True))

    def reproj_case():
        cams = _synthetic_cameras(lo, hi, rng)
        for reflect in (False, True):
            R = Rotation.from_rotvec([0.3, -0.5, 0.9]).as_matrix()
            if reflect:
                R = R @ np.diag([1.0, -1.0, 1.0])
            M = make_M(0.7, R, [1.0, 2.0, -0.5])
            check_similarity(M)
            check_reprojection_invariance(cams, M, X_box, tol=1e-6, rng=rng)

    report("reprojection invariance P' = P inv(M) (incl. reflection)", reproj_case)

    def frustum_case():
        cams = _synthetic_cameras(lo, hi, rng)
        M = make_M(0.7, Rotation.from_rotvec([0.3, -0.5, 0.9]).as_matrix()
                   @ np.diag([1.0, -1.0, 1.0]), [1.0, 2.0, -0.5])
        c_box = (lo + hi) / 2.0
        for cam in cams:
            Pp = transform_P(cam["P"], M)
            C = transform_points(M, camera_center(cam["P"])[None])[0]
            assert np.allclose(camera_center(Pp), C, atol=1e-8)
            corners = frustum_points(Pp, C, cam["resolution"], length=0.15)
            # frustum must open toward the (transformed) box center
            to_box = transform_points(M, c_box[None])[0] - C
            look = corners.mean(axis=0) - C
            assert look @ to_box > 0, "frustum points away from arena"

    report("camera centers + frusta (reflection-safe)", frustum_case)

    def xml_case():
        cams = _synthetic_cameras(lo, hi, rng)
        M = make_M(1.3, Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix(), [0.1, 0, -0.2])
        out_cams = [dict(c, P=transform_P(c["P"], M)) for c in cams]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test-aligned.xml")
            write_calibration_xml(out_cams, path)
            check_roundtrip(out_cams, path)

    report("calibration XML write/parse round trip", xml_case)

    def compose_case():
        M_fit = make_M(1.0, np.eye(3), [0.0, 0.0, 0.0])
        c_box = (lo + hi) / 2.0
        M = compose_manual(M_fit, c_box, flips=(-1, 1, 1))
        # flip about box center: center itself must not move
        assert np.allclose(transform_points(M, c_box[None])[0], c_box, atol=1e-12)
        check_similarity(M)

    report("manual compose (flip pivots about box center)", compose_case)

    print(f"\nselftest: {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURE(S)'}")
    return n_fail


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="run synthetic checks")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(1 if selftest() else 0)
    ap.print_help()
