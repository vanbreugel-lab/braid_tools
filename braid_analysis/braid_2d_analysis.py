"""
braid_2d_analysis.py

Helper functions for loading, reprojecting, and analyzing 2D and 3D tracking
data from braidz files combined with optogenetic trigger data.

Typical workflow
----------------
1. load_exp_code / load_triggers       -- read experiment config and trigger events
2. load_calibration                    -- parse camera projection matrices from braidz
3. build_trigger_frame_window          -- identify which frames fall near trigger events
4. load_2d_for_frames                  -- efficiently load only the needed 2D rows
5. reproject / find_triggered_obj_id   -- single-event inspection helpers
6. compute_3d_start_stop_times         -- meta-analysis: when does 3D tracking start/stop?
7. compute_2d_correspondence_times     -- meta-analysis: when is 2D tracking present?
8. compute_2d_correspondence_pixels    -- meta-analysis: where in the image is the fly?
9. plot_2d_pixel_locations             -- visualise per-camera pixel coverage
"""

import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import h5py

from braid_analysis import braid_filemanager

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')

def get_filename(data_folder, extension):
    """Wrapper around braid_filemanager.get_filename that strips ANSI color codes."""
    return _ANSI_ESCAPE.sub('', braid_filemanager.get_filename(data_folder, extension)).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Experiment configuration loading
# ──────────────────────────────────────────────────────────────────────────────

def load_exp_code(data_folder):
    """Load the experiment configuration YAML from the exp_code subdirectory.

    Parameters
    ----------
    data_folder : str
        Path to the experiment directory containing an ``exp_code/`` subdirectory
        with exactly one ``.yaml`` file.

    Returns
    -------
    exp_code : dict
        Parsed YAML contents. Commonly used keys include ``column_names``,
        ``topic``, ``xmin``/``xmax``/``ymin``/``ymax``/``zmin``/``zmax``
        (trigger volume bounds), and ``arduino_events``.
    """
    exp_code_dir = os.path.join(data_folder, 'exp_code')
    yaml_files = [f for f in os.listdir(exp_code_dir) if f.endswith('.yaml')]
    if not yaml_files:
        raise FileNotFoundError(f'No .yaml file found in {exp_code_dir}')
    with open(os.path.join(exp_code_dir, yaml_files[0])) as f:
        return yaml.safe_load(f)


def load_triggers(data_folder, exp_code):
    """Load the braid trigger HDF5 file and return a tidy DataFrame.

    The raw HDF5 dataset stores trigger metadata in ``data_0``…``data_N``
    columns.  This function renames the trailing columns to the human-readable
    names from ``exp_code['column_names']`` (e.g. ``first_flash_duration``).

    Parameters
    ----------
    data_folder : str
        Path to the experiment directory containing the ``.hdf5`` trigger file.
    exp_code : dict
        Parsed experiment YAML (see :func:`load_exp_code`).  Must contain the
        key ``column_names`` (list of str) and optionally ``topic`` (str,
        defaults to ``'braid_trigger_topic'``).

    Returns
    -------
    df_triggers : pandas.DataFrame
        One row per trigger event.  Columns include ``t`` (Unix timestamp),
        ``t_secs``, ``t_nsecs``, and the renamed arduino parameter columns.
    """
    trigger_column_names = exp_code['column_names']
    topic = exp_code.get('topic', 'braid_trigger_topic')

    hdf5_filename = get_filename(data_folder, '.hdf5')
    with h5py.File(hdf5_filename, 'r') as f:
        ds = f[topic]
        df = pd.DataFrame(ds[:])

    # Rename the trailing data_N columns to human-readable arduino parameter names.
    # The first n_meta data_N columns are tracking metadata (obj_id, frame, etc.).
    n_meta = len(df.columns) - 3 - len(trigger_column_names)  # 3 = t_secs, t_nsecs, t
    rename_map = {f'data_{n_meta + i}': trigger_column_names[i]
                  for i in range(len(trigger_column_names))}
    return df.rename(columns=rename_map)


# ──────────────────────────────────────────────────────────────────────────────
# Braidz data loading
# ──────────────────────────────────────────────────────────────────────────────

def build_trigger_frame_window(df_triggers, df_3d, window_s=5.0):
    """Return the set of 3D frame numbers that fall within ±window_s of any trigger.

    Uses the ``timestamp`` column of ``df_3d`` to convert trigger times to
    frame numbers, so no fixed frame-rate assumption is required.

    Parameters
    ----------
    df_triggers : pandas.DataFrame
        Trigger event table with a ``t`` column (Unix timestamps).
    df_3d : pandas.DataFrame
        Full 3D Kalman-filter dataframe with ``timestamp`` and ``frame`` columns.
    window_s : float
        Half-width of the time window around each trigger (seconds).

    Returns
    -------
    wanted_frames : set of int
        Frame numbers within the time windows.  Pass directly to
        :func:`load_2d_for_frames`.
    """
    wanted_frames = set()
    for t_trigger in df_triggers.t.values:
        mask = ((df_3d.timestamp >= t_trigger - window_s) &
                (df_3d.timestamp <= t_trigger + window_s))
        wanted_frames.update(df_3d.loc[mask, 'frame'].values)
    return wanted_frames


def load_2d_for_frames(filename_or_url, wanted_frames, chunksize=500_000):
    """Load 2D tracking data from a braidz file for a specific set of frames only.

    Reading ``data2d_distorted.csv.gz`` in full can take a very long time.
    This function reads it in chunks and discards any row whose frame number
    is not in ``wanted_frames``, so only the data you need is kept in memory.

    Parameters
    ----------
    filename_or_url : str
        Path (or URL) to the ``.braidz`` file.
    wanted_frames : set or array-like of int
        Frame numbers to retain.
    chunksize : int
        Number of CSV rows to read per chunk.  Larger values are faster but
        use more memory.

    Returns
    -------
    df_2d : pandas.DataFrame
        2D detections for the requested frames.  Columns include ``camn``,
        ``frame``, ``timestamp``, ``x``, ``y``, ``area``, etc.
        Returns an empty DataFrame if no matching rows are found.
    """
    fileobj = braid_filemanager.open_filename_or_url(filename_or_url)
    chunks = []
    with zipfile.ZipFile(file=fileobj, mode='r') as archive:
        reader = pd.read_csv(
            archive.open('data2d_distorted.csv.gz'),
            comment='#',
            compression='gzip',
            chunksize=chunksize,
        )
        for chunk in reader:
            filtered = chunk[chunk['frame'].isin(wanted_frames)]
            if len(filtered) > 0:
                chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Calibration and reprojection
# ──────────────────────────────────────────────────────────────────────────────

def load_calibration(braidz_filename):
    """Parse camera calibration from a braidz file.

    Reads ``calibration.xml`` (3×4 projection matrices and image resolutions)
    and ``cam_info.csv.gz`` (camn ↔ cam_id mapping) from inside the braidz
    ZIP archive.

    Parameters
    ----------
    braidz_filename : str
        Path (or URL) to the ``.braidz`` file.

    Returns
    -------
    camn_to_P : dict
        Maps integer ``camn`` to a (3, 4) numpy array — the linear projection
        matrix for that camera.
    camn_to_id : dict
        Maps integer ``camn`` to the camera ID string (e.g. ``'Basler-40196688'``).
    camn_to_res : dict
        Maps integer ``camn`` to a ``(width, height)`` tuple of the image
        resolution in pixels.

    Notes
    -----
    The projection matrices map 3D homogeneous world coordinates
    ``[X, Y, Z, 1]`` to undistorted pixel coordinates.  ``data2d_distorted``
    uses distorted pixel coordinates, so small systematic offsets (primarily
    near image edges) are expected when comparing reprojected positions to
    measured 2D detections.
    """
    fileobj = braid_filemanager.open_filename_or_url(braidz_filename)
    with zipfile.ZipFile(file=fileobj, mode='r') as archive:
        tree = ET.parse(archive.open('calibration.xml'))
        root = tree.getroot()
        camid_to_P = {}
        camid_to_res = {}
        for cam in root.findall('single_camera_calibration'):
            cam_id = cam.find('cam_id').text
            mat_str = cam.find('calibration_matrix').text
            rows = mat_str.strip().split(';')
            P = np.array([[float(v) for v in row.split()] for row in rows])
            camid_to_P[cam_id] = P
            w, h = cam.find('resolution').text.split()
            camid_to_res[cam_id] = (int(w), int(h))
        cam_info = pd.read_csv(
            archive.open('cam_info.csv.gz'), compression='gzip', comment='#')

    camn_to_P = {int(r.camn): camid_to_P[r.cam_id]
                 for _, r in cam_info.iterrows() if r.cam_id in camid_to_P}
    camn_to_id = {int(r.camn): r.cam_id for _, r in cam_info.iterrows()}
    camn_to_res = {int(r.camn): camid_to_res[r.cam_id]
                   for _, r in cam_info.iterrows() if r.cam_id in camid_to_res}
    return camn_to_P, camn_to_id, camn_to_res


def reproject(P, xyz):
    """Project an array of 3D world points into a camera's pixel space.

    Parameters
    ----------
    P : numpy.ndarray, shape (3, 4)
        Linear camera projection matrix (from :func:`load_calibration`).
    xyz : numpy.ndarray, shape (N, 3)
        3D world coordinates to project.

    Returns
    -------
    u : numpy.ndarray, shape (N,)
        Horizontal pixel coordinates (undistorted).
    v : numpy.ndarray, shape (N,)
        Vertical pixel coordinates (undistorted).
    """
    pts_h = np.hstack([xyz, np.ones((len(xyz), 1))])  # Nx4 homogeneous
    proj = P @ pts_h.T                                 # 3xN
    u = proj[0] / proj[2]
    v = proj[1] / proj[2]
    return u, v


# ──────────────────────────────────────────────────────────────────────────────
# Trigger-object identification
# ──────────────────────────────────────────────────────────────────────────────

def find_triggered_obj_id(t_trigger, df_3d, exp_code, dt_search=0.3):
    """Find the tracked object that was inside the trigger volume at a given time.

    Searches ``df_3d`` for rows within ±``dt_search`` seconds of ``t_trigger``
    whose (x, y, z) position falls within the bounding box defined by the
    ``xmin``/``xmax``/``ymin``/``ymax``/``zmin``/``zmax`` fields of
    ``exp_code``.

    Parameters
    ----------
    t_trigger : float
        Unix timestamp of the trigger event.
    df_3d : pandas.DataFrame
        Full 3D Kalman-filter dataframe with columns ``timestamp``, ``obj_id``,
        ``x``, ``y``, ``z``.
    exp_code : dict
        Parsed experiment YAML containing the trigger volume bounds.
    dt_search : float
        Half-width (seconds) of the time window used to search for the object.

    Returns
    -------
    obj_id : int or None
        The ``obj_id`` of the first object found in the trigger volume, or
        ``None`` if no object was found (e.g. sham events).
    """
    nearby = df_3d[
        (df_3d.timestamp >= t_trigger - dt_search) &
        (df_3d.timestamp <= t_trigger + dt_search)
    ]
    in_zone = nearby[
        (nearby.x >= exp_code['xmin']) & (nearby.x <= exp_code['xmax']) &
        (nearby.y >= exp_code['ymin']) & (nearby.y <= exp_code['ymax']) &
        (nearby.z >= exp_code['zmin']) & (nearby.z <= exp_code['zmax'])
    ]
    if len(in_zone) == 0:
        return None
    return in_zone.obj_id.values[0]


# ──────────────────────────────────────────────────────────────────────────────
# Correspondence matching (internal helper)
# ──────────────────────────────────────────────────────────────────────────────

def _correspondence_matches(traj, cam_2d, P, res, distance_threshold):
    """Return a boolean mask of 2D detections that correspond to a 3D trajectory.

    For each row in ``cam_2d``, finds the temporally closest frame in ``traj``,
    reprojects that 3D position into the camera, and checks whether the pixel
    distance to the 2D detection is below ``distance_threshold``.  Matches
    where the reprojected position falls outside the image bounds are rejected
    regardless of pixel distance.

    Parameters
    ----------
    traj : pandas.DataFrame
        3D trajectory slice (sorted by ``timestamp``) with columns
        ``timestamp``, ``x``, ``y``, ``z``.
    cam_2d : pandas.DataFrame
        2D detection rows for a single camera (NaN ``x`` rows already dropped),
        with columns ``timestamp``, ``x``, ``y``.
    P : numpy.ndarray, shape (3, 4)
        Camera projection matrix.
    res : tuple of (int, int)
        Image ``(width, height)`` used for bounds checking.
    distance_threshold : float
        Maximum pixel distance (in undistorted space) between a 2D detection
        and the reprojected 3D position for the pair to be counted as a match.

    Returns
    -------
    mask : numpy.ndarray of bool, shape (len(cam_2d),)
        ``True`` for rows that are in-bounds and within ``distance_threshold``.
    """
    t_3d = traj['timestamp'].values
    xyz  = traj[['x', 'y', 'z']].values
    t_2d = cam_2d['timestamp'].values
    w, h = res

    u_proj, v_proj = reproject(P, xyz)

    # For each 2D detection find the temporally closest reprojected 3D position
    idx = np.searchsorted(t_3d, t_2d).clip(0, len(t_3d) - 1)
    idx_prev = np.maximum(idx - 1, 0)
    closer_prev = np.abs(t_3d[idx_prev] - t_2d) < np.abs(t_3d[idx] - t_2d)
    idx = np.where(closer_prev, idx_prev, idx)

    in_bounds = (
        (u_proj[idx] >= 0) & (u_proj[idx] < w) &
        (v_proj[idx] >= 0) & (v_proj[idx] < h)
    )
    dist = np.sqrt(
        (cam_2d['x'].values - u_proj[idx]) ** 2 +
        (cam_2d['y'].values - v_proj[idx]) ** 2
    )
    return in_bounds & (dist < distance_threshold)


# ──────────────────────────────────────────────────────────────────────────────
# Meta-analysis across all trigger events
# ──────────────────────────────────────────────────────────────────────────────

def compute_3d_start_stop_times(df_triggers, df_3d, exp_code,
                                 window_s=5.0, n_triggers=None):
    """Compute when the triggered object's 3D trajectory starts and stops
    relative to each trigger, across all trigger events.

    Parameters
    ----------
    df_triggers : pandas.DataFrame
        Trigger event table (see :func:`load_triggers`).
    df_3d : pandas.DataFrame
        Full 3D Kalman-filter dataframe.
    exp_code : dict
        Parsed experiment YAML (used by :func:`find_triggered_obj_id`).
    window_s : float
        Half-width of the time window to search for the trajectory (seconds).
    n_triggers : int or None
        If given, analyse only the first ``n_triggers`` rows of ``df_triggers``.
        Useful for quick tests before running the full dataset.

    Returns
    -------
    start_times : list of float
        Time (seconds relative to trigger) of the first 3D frame found in the
        window for each trigger event where an object was detected.
    stop_times : list of float
        Time (seconds relative to trigger) of the last 3D frame found in the
        window for each trigger event.
    """
    triggers = df_triggers.iloc[:n_triggers] if n_triggers is not None else df_triggers
    start_times, stop_times = [], []

    for _, trig in triggers.iterrows():
        t_trig = trig['t']
        obj_id = find_triggered_obj_id(t_trig, df_3d, exp_code)
        if obj_id is None:
            continue
        traj = df_3d[
            (df_3d.obj_id == obj_id) &
            (df_3d.timestamp >= t_trig - window_s) &
            (df_3d.timestamp <= t_trig + window_s)
        ]
        if len(traj) < 2:
            continue
        t_rel = traj['timestamp'].values - t_trig
        start_times.append(float(t_rel.min()))
        stop_times.append(float(t_rel.max()))

    print(f'compute_3d_start_stop_times: {len(start_times)} trajectories found '
          f'(of {len(triggers)} triggers)')
    return start_times, stop_times


def compute_2d_correspondence_times(df_triggers, df_3d, df_2d,
                                     camn_to_P, camn_to_res, exp_code,
                                     window_s=5.0, distance_threshold=30,
                                     n_triggers=None):
    """Collect times of 2D detections that correspond to the triggered 3D
    trajectory, per camera, across all trigger events.

    For each trigger event and each camera, 2D detections are matched to the
    3D trajectory using :func:`_correspondence_matches`.  Only detections whose
    nearest reprojected position is within ``distance_threshold`` pixels *and*
    within image bounds are included.

    Parameters
    ----------
    df_triggers : pandas.DataFrame
        Trigger event table.
    df_3d : pandas.DataFrame
        Full 3D Kalman-filter dataframe.
    df_2d : pandas.DataFrame
        2D detections pre-filtered to the relevant time windows (e.g. from
        :func:`load_2d_for_frames`).
    camn_to_P : dict
        Camera projection matrices (from :func:`load_calibration`).
    camn_to_res : dict
        Camera resolutions (from :func:`load_calibration`).
    exp_code : dict
        Parsed experiment YAML.
    window_s : float
        Half-width of the time window around each trigger (seconds).
    distance_threshold : float
        Maximum pixel distance for a 2D detection to be counted as a
        correspondence with the reprojected 3D trajectory.
    n_triggers : int or None
        If given, analyse only the first ``n_triggers`` rows.

    Returns
    -------
    corr_times : dict
        Maps each ``camn`` (int) to a list of floats — the times in seconds
        relative to the trigger at which a corresponding 2D detection was found.
    """
    triggers = df_triggers.iloc[:n_triggers] if n_triggers is not None else df_triggers
    corr_times = {camn: [] for camn in camn_to_P}

    for _, trig in triggers.iterrows():
        t_trig = trig['t']
        obj_id = find_triggered_obj_id(t_trig, df_3d, exp_code)
        if obj_id is None:
            continue
        traj = df_3d[
            (df_3d.obj_id == obj_id) &
            (df_3d.timestamp >= t_trig - window_s) &
            (df_3d.timestamp <= t_trig + window_s)
        ].sort_values('timestamp')
        if len(traj) < 2:
            continue
        df_2d_win = df_2d[
            (df_2d.timestamp >= t_trig - window_s) &
            (df_2d.timestamp <= t_trig + window_s)
        ]
        for camn, P in camn_to_P.items():
            cam_2d = df_2d_win[df_2d_win['camn'] == camn].dropna(subset=['x'])
            if len(cam_2d) == 0:
                continue
            mask = _correspondence_matches(
                traj, cam_2d, P, camn_to_res.get(camn, (1920, 1200)),
                distance_threshold)
            corr_times[camn].extend(
                (cam_2d['timestamp'].values[mask] - t_trig).tolist())

    print(f'compute_2d_correspondence_times: processed {len(triggers)} triggers')
    return corr_times


def compute_2d_correspondence_pixels(df_triggers, df_3d, df_2d,
                                      camn_to_P, camn_to_res, exp_code,
                                      window_s=5.0, distance_threshold=30,
                                      n_triggers=None):
    """Collect pixel coordinates of 2D detections that correspond to the
    triggered 3D trajectory, per camera, across all trigger events.

    In addition to the matched 2D pixel positions, also stores the reprojected
    3D position at the trigger moment (t_rel = 0) for each event and camera.
    This is used by :func:`plot_2d_pixel_locations` to draw a red marker showing
    where the fly was at the moment it triggered the stimulus.  If the trigger-
    moment reprojection falls outside image bounds, no marker is stored for that
    camera/event combination.

    Parameters
    ----------
    df_triggers : pandas.DataFrame
        Trigger event table.
    df_3d : pandas.DataFrame
        Full 3D Kalman-filter dataframe.
    df_2d : pandas.DataFrame
        2D detections pre-filtered to the relevant time windows.
    camn_to_P : dict
        Camera projection matrices (from :func:`load_calibration`).
    camn_to_res : dict
        Camera resolutions (from :func:`load_calibration`).
    exp_code : dict
        Parsed experiment YAML.
    window_s : float
        Half-width of the time window around each trigger (seconds).
    distance_threshold : float
        Maximum pixel distance for a 2D detection to be counted as a
        correspondence.
    n_triggers : int or None
        If given, analyse only the first ``n_triggers`` rows.

    Returns
    -------
    corr_pixels : dict
        Maps each ``camn`` (int) to a dict with keys:

        ``'x'``, ``'y'``
            Pixel coordinates of matched 2D detections.
        ``'t_rel'``
            Time relative to trigger (seconds) for each matched detection.
        ``'event_idx'``
            Integer index of the trigger event each detection belongs to,
            allowing per-event grouping in downstream analysis.
        ``'trigger_u'``, ``'trigger_v'``
            Reprojected pixel coordinates of the 3D position at trigger time,
            one entry per event where the projection is within image bounds.
    """
    triggers = df_triggers.iloc[:n_triggers] if n_triggers is not None else df_triggers
    corr_pixels = {camn: {'x': [], 'y': [], 't_rel': [], 'event_idx': [],
                           'trigger_u': [], 'trigger_v': []}
                   for camn in camn_to_P}

    for event_idx, (_, trig) in enumerate(triggers.iterrows()):
        t_trig = trig['t']
        obj_id = find_triggered_obj_id(t_trig, df_3d, exp_code)
        if obj_id is None:
            continue
        traj = df_3d[
            (df_3d.obj_id == obj_id) &
            (df_3d.timestamp >= t_trig - window_s) &
            (df_3d.timestamp <= t_trig + window_s)
        ].sort_values('timestamp')
        if len(traj) < 2:
            continue

        t_3d = traj['timestamp'].values
        xyz  = traj[['x', 'y', 'z']].values
        trig_frame_idx = np.argmin(np.abs(t_3d - t_trig))
        xyz_at_trig = xyz[trig_frame_idx:trig_frame_idx + 1]

        df_2d_win = df_2d[
            (df_2d.timestamp >= t_trig - window_s) &
            (df_2d.timestamp <= t_trig + window_s)
        ]

        for camn, P in camn_to_P.items():
            w, h = camn_to_res.get(camn, (1920, 1200))

            # Reprojected trigger-moment position (red dot in plot)
            u_t, v_t = reproject(P, xyz_at_trig)
            if 0 <= u_t[0] < w and 0 <= v_t[0] < h:
                corr_pixels[camn]['trigger_u'].append(float(u_t[0]))
                corr_pixels[camn]['trigger_v'].append(float(v_t[0]))

            # Matched 2D detections across the full window
            cam_2d = df_2d_win[df_2d_win['camn'] == camn].dropna(subset=['x'])
            if len(cam_2d) == 0:
                continue
            mask = _correspondence_matches(traj, cam_2d, P, (w, h),
                                           distance_threshold)
            n = mask.sum()
            corr_pixels[camn]['x'].extend(cam_2d['x'].values[mask].tolist())
            corr_pixels[camn]['y'].extend(cam_2d['y'].values[mask].tolist())
            corr_pixels[camn]['t_rel'].extend(
                (cam_2d['timestamp'].values[mask] - t_trig).tolist())
            corr_pixels[camn]['event_idx'].extend([event_idx] * n)

    print(f'compute_2d_correspondence_pixels: done ({len(triggers)} triggers)')
    return corr_pixels


# ──────────────────────────────────────────────────────────────────────────────
# Reprojection helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_reprojections_for_trigger(t_trigger, df_3d, camn_to_P, window_s=5.0):
    """Reproject every 3D object visible near a trigger event into each camera.

    Parameters
    ----------
    t_trigger : float
        Unix timestamp of the trigger event.
    df_3d : pandas.DataFrame
        Full 3D Kalman-filter dataframe with columns ``timestamp``, ``obj_id``,
        ``x``, ``y``, ``z``.
    camn_to_P : dict
        Maps ``camn`` (int) to a (3, 4) projection matrix.
    window_s : float
        Half-width of the time window to search (seconds).

    Returns
    -------
    reprojections : dict
        Nested dict ``{obj_id: {camn: (t_rel, u, v)}}`` where ``t_rel`` is a
        1-D array of times relative to the trigger and ``u``, ``v`` are pixel
        coordinate arrays of the same length.
    """
    all_obj_ids = df_3d[
        (df_3d.timestamp >= t_trigger - window_s) &
        (df_3d.timestamp <= t_trigger + window_s)
    ].obj_id.unique()

    reprojections = {}
    for oid in all_obj_ids:
        traj = df_3d[
            (df_3d.obj_id == oid) &
            (df_3d.timestamp >= t_trigger - window_s) &
            (df_3d.timestamp <= t_trigger + window_s)
        ]
        xyz   = traj[['x', 'y', 'z']].values
        t_rel = traj['timestamp'].values - t_trigger
        reprojections[oid] = {}
        for camn, P in camn_to_P.items():
            u, v = reproject(P, xyz)
            reprojections[oid][camn] = (t_rel, u, v)

    return reprojections


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_single_trigger_event(trigger_idx, df_triggers, df_3d, df_2d_windowed,
                               exp_code, camn_to_P, camn_to_id,
                               window_s=5.0, ncols=3):
    """Plot the 3D trajectory and 2D detections for a single trigger event.

    Produces two figures:

    1. **Two-panel overview** — top panel shows the 3D x-position of the
       triggered object over time; bottom panel shows x-pixel positions from
       all cameras for all 2D detections in the window.

    2. **Per-camera reprojection grid** — one subplot per camera that had 2D
       detections, showing measured 2D x-pixel (blue dots) and the reprojected
       x-pixel of the triggered object (orange line) and all other 3D objects
       (grey lines) in the window.

    Parameters
    ----------
    trigger_idx : int
        Index into ``df_triggers`` (0-based) of the event to inspect.
    df_triggers : pandas.DataFrame
        Trigger event table (see :func:`load_triggers`).
    df_3d : pandas.DataFrame
        Full 3D Kalman-filter dataframe.
    df_2d_windowed : pandas.DataFrame
        2D detections pre-filtered to the relevant time windows (from
        :func:`load_2d_for_frames`).
    exp_code : dict
        Parsed experiment YAML (used to locate the triggered object).
    camn_to_P : dict
        Camera projection matrices (from :func:`load_calibration`).
    camn_to_id : dict
        Maps ``camn`` (int) to camera ID string (from :func:`load_calibration`).
    window_s : float
        Half-width of the time window to display (seconds).
    ncols : int
        Number of columns in the per-camera reprojection grid.
    """
    trigger   = df_triggers.iloc[trigger_idx]
    t_trigger = trigger['t']
    triggered_obj_id = find_triggered_obj_id(t_trigger, df_3d, exp_code)

    print(f'Trigger #{trigger_idx}  t={t_trigger:.1f}  obj_id={triggered_obj_id}')
    print(trigger.to_string())

    # ── Slice data to the trigger window ──────────────────────────────────────
    df_3d_trig = df_3d[
        (df_3d.obj_id == triggered_obj_id) &
        (df_3d.timestamp >= t_trigger - window_s) &
        (df_3d.timestamp <= t_trigger + window_s)
    ].copy()
    df_3d_trig['t_rel'] = df_3d_trig['timestamp'] - t_trigger

    df_2d_trig = df_2d_windowed[
        (df_2d_windowed.timestamp >= t_trigger - window_s) &
        (df_2d_windowed.timestamp <= t_trigger + window_s)
    ].copy()
    df_2d_trig['t_rel'] = df_2d_trig['timestamp'] - t_trigger
    df_2d_trig = df_2d_trig.dropna(subset=['x'])

    # ── Figure 1: two-panel overview ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(df_3d_trig['t_rel'], df_3d_trig['x'], color='steelblue', linewidth=1.5)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1, label='trigger')
    ax1.set_ylabel('x position (m)')
    ax1.set_title(f'3D trajectory  |  obj_id={triggered_obj_id}  |  trigger #{trigger_idx}')
    ax1.legend()

    camns = sorted(df_2d_trig.camn.unique())
    cmap  = plt.cm.tab20
    for i, camn in enumerate(camns):
        sub = df_2d_trig[df_2d_trig.camn == camn]
        ax2.scatter(sub['t_rel'], sub['x'], s=3,
                    color=cmap(i / max(len(camns), 1)),
                    label=camn_to_id.get(camn, str(camn)), alpha=0.7, rasterized=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel('x pixel')
    ax2.set_xlabel('time relative to trigger (s)')
    ax2.set_title('2D detections per camera (all objects)')
    ax2.legend(markerscale=3, fontsize=7, ncol=4, loc='upper right')

    plt.tight_layout()
    plt.show()

    # ── Reproject all 3D objects into each camera ─────────────────────────────
    reprojections_all = compute_reprojections_for_trigger(
        t_trigger, df_3d, camn_to_P, window_s=window_s)

    all_obj_ids   = list(reprojections_all.keys())
    other_obj_ids = [oid for oid in all_obj_ids if oid != triggered_obj_id]

    # ── Figure 2: per-camera reprojection grid ────────────────────────────────
    camns_with_detections = sorted(df_2d_trig.camn.unique())
    nrows = int(np.ceil(len(camns_with_detections) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True)
    axes = np.array(axes).flatten()

    for i, camn in enumerate(camns_with_detections):
        ax  = axes[i]
        sub = df_2d_trig[df_2d_trig.camn == camn]
        ax.scatter(sub['t_rel'], sub['x'], s=4, color='steelblue',
                   label='2D detected', zorder=2, alpha=0.7)
        for oid in other_obj_ids:
            if camn in reprojections_all.get(oid, {}):
                t_r, u, v = reprojections_all[oid][camn]
                ax.plot(t_r, u, color='grey', linewidth=0.8, alpha=0.5,
                        label='other obj' if oid == other_obj_ids[0] else '_')
        if triggered_obj_id is not None and camn in reprojections_all.get(triggered_obj_id, {}):
            t_r, u, v = reprojections_all[triggered_obj_id][camn]
            ax.plot(t_r, u, color='orange', linewidth=2.0, label='triggered obj', zorder=4)
        ax.axvline(0, color='red', linestyle='--', linewidth=0.8)
        ax.set_title(camn_to_id.get(camn, f'cam {camn}'), fontsize=9)
        ax.set_ylabel('x pixel', fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(camns_with_detections):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(camns_with_detections) - ncols):len(camns_with_detections)]:
        ax.set_xlabel('time rel. to trigger (s)', fontsize=8)

    seen = {}
    for h, l in zip(*axes[0].get_legend_handles_labels()):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc='lower right', fontsize=9)
    fig.suptitle(
        f'Reprojected 3D vs 2D — trigger #{trigger_idx}, obj_id={triggered_obj_id}'
        f'  ({len(all_obj_ids)} objects in window)',
        fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_3d_start_stop_histogram(start_times, stop_times, window_s, bins=None):
    """Plot a histogram of 3D trajectory start and stop times relative to trigger.

    Parameters
    ----------
    start_times : list of float
        Times (seconds relative to trigger) when each trajectory begins in the
        window; output of :func:`compute_3d_start_stop_times`.
    stop_times : list of float
        Times (seconds relative to trigger) when each trajectory ends.
    window_s : float
        Half-width of the time window; used to set the default bin range.
    bins : array-like or int or None
        Histogram bins passed to ``numpy.histogram``.  Defaults to 60 equal
        bins spanning ``[-window_s, window_s]``.
    """
    if bins is None:
        bins = np.linspace(-window_s, window_s, 60)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(start_times, bins=bins, color='green', alpha=0.7, label='3D start')
    ax.hist(stop_times,  bins=bins, color='red',   alpha=0.7, label='3D stop')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='trigger')
    ax.set_xlabel('time relative to trigger (s)')
    ax.set_ylabel('count')
    ax.set_title(f'3D trajectory start / stop times  (n={len(start_times)} trajectories)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_2d_correspondence_histograms(corr_times, camn_to_id, window_s,
                                       distance_threshold, bins=None, ncols=3):
    """Plot per-camera histograms of 2D detection times relative to trigger.

    Each subplot shows, for one camera, when (relative to each trigger) a 2D
    detection was found within ``distance_threshold`` pixels of the reprojected
    3D trajectory.

    Parameters
    ----------
    corr_times : dict
        Output of :func:`compute_2d_correspondence_times`.  Maps ``camn`` (int)
        to a list of floats (seconds relative to trigger).
    camn_to_id : dict
        Maps ``camn`` (int) to camera ID string; used as subplot titles.
    window_s : float
        Half-width of the time window; used to set the default bin range.
    distance_threshold : float
        Pixel threshold used when computing ``corr_times``; shown in the figure
        title for reference.
    bins : array-like or int or None
        Histogram bins.  Defaults to 60 equal bins spanning
        ``[-window_s, window_s]``.
    ncols : int
        Number of columns in the subplot grid.
    """
    if bins is None:
        bins = np.linspace(-window_s, window_s, 60)

    n_triggers_approx = max(
        (len(v) for v in corr_times.values() if v), default=0)

    camns_active = [c for c in sorted(corr_times) if len(corr_times[c]) > 0]
    nrows = int(np.ceil(len(camns_active) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for i, camn in enumerate(camns_active):
        axes[i].hist(corr_times[camn], bins=bins, color='steelblue', alpha=0.8)
        axes[i].axvline(0, color='red', linestyle='--', linewidth=0.8)
        axes[i].set_title(camn_to_id.get(camn, f'cam {camn}'), fontsize=9)
        axes[i].set_ylabel('count', fontsize=8)
        axes[i].tick_params(labelsize=7)

    for ax in axes[len(camns_active):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(camns_active) - ncols):len(camns_active)]:
        ax.set_xlabel('time rel. to trigger (s)', fontsize=8)

    fig.suptitle(
        f'2D detections corresponding to 3D trajectory  '
        f'(threshold={distance_threshold} px)',
        fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_2d_pixel_locations(corr_pixels, camn_to_id, ncols=3,
                             alpha=0.1, point_size=2):
    """Plot 2D pixel locations of trajectory correspondences, one panel per camera.

    Blue dots show all 2D detections matched to the 3D trajectory across the
    full time window.  Red dots show the reprojected 3D position at the trigger
    moment (one per event), giving a reference for where the fly was when it
    triggered the stimulus.  Red dots are only shown for cameras where the
    trigger-moment projection falls within image bounds.

    The y-axis is inverted to match image coordinates (origin at top-left).

    Parameters
    ----------
    corr_pixels : dict
        Output of :func:`compute_2d_correspondence_pixels`.
    camn_to_id : dict
        Maps ``camn`` (int) to camera ID string (from :func:`load_calibration`).
        Used as subplot titles.
    ncols : int
        Number of columns in the subplot grid.
    alpha : float
        Opacity of the blue scatter dots (0–1).  Reduce if the image is too dense.
    point_size : float
        Marker size (``s`` parameter) for the blue scatter dots.
    """
    camns_active = [c for c in sorted(corr_pixels)
                    if len(corr_pixels[c]['x']) > 0
                    or len(corr_pixels[c]['trigger_u']) > 0]
    nrows = int(np.ceil(len(camns_active) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, camn in enumerate(camns_active):
        ax = axes[i]
        x  = np.array(corr_pixels[camn]['x'])
        y  = np.array(corr_pixels[camn]['y'])
        tu = np.array(corr_pixels[camn]['trigger_u'])
        tv = np.array(corr_pixels[camn]['trigger_v'])

        if len(x):
            ax.scatter(x, y, s=point_size, alpha=alpha, color='steelblue',
                       rasterized=True)
        if len(tu):
            ax.scatter(tu, tv, s=60, color='red', zorder=5)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(camn_to_id.get(camn, f'cam {camn}'), fontsize=9)
        ax.set_xlabel('x pixel', fontsize=8)
        ax.set_ylabel('y pixel', fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(camns_active):]:
        ax.set_visible(False)

    handles = [
        plt.scatter([], [], s=point_size * 6, color='steelblue', alpha=0.6,
                    label='2D detections (trajectory window)'),
        plt.scatter([], [], s=60, color='red',
                    label='reprojected 3D position at trigger time'),
    ]
    fig.legend(handles=handles, loc='lower right', fontsize=9)
    fig.suptitle(
        f'2D pixel locations of trajectory correspondences  '
        f'(n={sum(len(v["x"]) for v in corr_pixels.values())} total detections)',
        fontsize=11)
    plt.tight_layout()
    plt.show()
