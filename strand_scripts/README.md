# strand_scripts

Standalone (ROS-free) tools that drive a local [strand-cam](https://github.com/strawlab/strand-braid)
camera directly. Require the strand-braid `.deb` (provides `strand-cam` and
`show-timestamps`) and system python3 with numpy/scipy/pandas/matplotlib/cv2/pyyaml.

| script | purpose |
|---|---|
| `collect_led_calibration_data.py` | record braid 3D + aux-camera 2D LED detections into a dataset dir |
| `led_calibration_merge.py` | merge/align the two streams (runs automatically; importable) |
| `calibrate_aux_camera.py` | fit K + lens distortion + R/t from a merged dataset |
| `dump_detection_config.py` | print a running strand-cam's detection settings as paste-ready toml |
| `led_calibration.toml` | config template for the collection script (lives right here) |

See the main repo `README.md` ("Standalone strand-cam tools") for background on
each script; this file is the hands-on workflow.

## 1. Collect a dataset

The config is `strand_scripts/led_calibration.toml` — co-located with the script
and used automatically when `--config` is not given. Edits take effect on the
next run (nothing is installed elsewhere). The `[[cameras]]` +
`[cameras.point_detection_config]` tables use braid's exact toml format.

```bash
cd ~/ros2_ws/src/braid_tools
python3 strand_scripts/collect_led_calibration_data.py
```

The script launches strand-cam and prints its browser UI URL. Put the LED in
view, check the live image shows the detection marker tracking it (tune in the
UI if needed; afterwards run `dump_detection_config.py` to copy tuned values
back into the toml — toml keys override UI-tuned values on the next launch).
Press Enter to start recording, then wave the LED through the whole tracked
volume for 1–2 minutes with continuously varying speed — cover as much of the
camera's field of view (and depth) as you can; a stationary LED cannot be
aligned. Ctrl-C stops, files the data, and runs the merge automatically.

Each session creates its own directory under `output_base_dir` (default
`~/Desktop/led_calibration_datasets/`):

```bash
ls -t ~/Desktop/led_calibration_datasets/    # newest led_calib_YYYYMMDD_HHMMSS is yours
```

## 2. Check the merge quality

Open `alignment_diagnostics.png` in the dataset dir and look at the
`alignment:` block of `metadata.yaml`. Healthy numbers:

- `match_method: frame` (exact triggerbox-frame pairing) with
  `frame_offset_consistency` ≈ 1.0
- `residual_rms_ms` well under 5 ms (half a frame at 100 fps)
- `n_matched` in the thousands, `merge_error: ''`

To re-run the merge on an existing dataset (after a failure or parameter tweak):

```bash
python3 strand_scripts/collect_led_calibration_data.py --merge-only \
    ~/Desktop/led_calibration_datasets/led_calib_<timestamp>
```

## 3. Run the calibration

```bash
python3 strand_scripts/calibrate_aux_camera.py \
    ~/Desktop/led_calibration_datasets/led_calib_<timestamp> --cross-check
```

`--cross-check` is optional: it prints how much an independent
`cv2.calibrateCamera` fit disagrees with the primary fit (small deltas =
trustworthy).

The fit is initialized with a RANSAC DLT, so heavy contamination from LED
*reflections* (off the arena floor/walls — spatially coherent streaks of bogus
detections, up to ~45% of rows) is rejected automatically. `--ransac-px`
adjusts the consensus threshold (default 8 px).

Healthy signs in the printed summary (benchmark from the first real run):

- `full_model` median ~1–3 px, clearly better than `linear_dlt` (~5 px)
- `fx` ≈ `fy` within a few percent; `cx, cy` near the image center (960, 600)
- outliers can be a large fraction if the arena reflects the LED — fine as
  long as the RANSAC consensus is a solid majority and the kept-inlier
  medians look good; check the rejected points form reflection-shaped
  streaks in the diagnostics rather than random scatter
- **no** "parameter(s) sit on their bound" warnings (that means an ill-posed fit)

Then inspect `calibration_diagnostics.png`: the left quiver (linear DLT) shows
large radially-organized residual arrows that should mostly vanish in the
middle quiver (full model), and in the bottom panel the red reprojected
trajectory should hug the cyan detections.

Tuning knobs if the fit looks off: `--outlier-px 4` forces a stricter rejection
threshold; `--k3` frees the r⁶ radial term (only sensible if the LED covered
most of the sensor — check `valid_pixel_region` in the yaml, the distortion
model is extrapolated outside it).

## 4. The product

`camera_calibration.yaml` in the dataset dir: K, distortion (OpenCV plumb_bob),
R/t, and `P_linear` (drop-in for `braid_analysis.braid_2d_analysis.reproject`).
Downstream code loads it with:

```python
from calibrate_aux_camera import load_camera_calibration, project_points
calib = load_camera_calibration('.../camera_calibration.yaml')
u, v, in_front = project_points(xyz, calib)   # full distortion model
```

Datasets are self-contained — you can recollect and recalibrate any time and
compare the resulting yamls side by side.

## Selftests (no hardware needed)

```bash
python3 strand_scripts/led_calibration_merge.py --selftest
python3 strand_scripts/calibrate_aux_camera.py --selftest
```
