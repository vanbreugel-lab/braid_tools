# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`braid_tools` is a ROS 2 package (this `ros2` branch targets Jazzy/Ubuntu 24.04; the `main` branch holds the original ROS 1 catkin version) with two distinct halves:

1. **ROS nodes** (`nodes/`, `msg/`, `launch/`) — bridge a running [Braid](https://github.com/strawlab/strand-braid) fly-tracking server to ROS, record its output, and visualize it live.
2. **`braid_analysis` Python package** (`braid_analysis/`) — offline analysis library for `.braidz` archives (Braid's output format), independent of ROS. This half is installed separately via `setup.py` and is what most data-analysis work touches. It is identical between the `main` and `ros2` branches — analysis changes belong on `main` (or should be kept in sync).

Data flows: Braid HTTP event stream → `braid_ros_listener.py` → ROS topic `flydra_mainbrain/super_packets` → either `braid_realtime_plotter.py` (live viz) or `braid_save_data_to_hdf5.py` (recording). Offline, a `.braidz` archive is loaded and processed by `braid_analysis` without any ROS dependency.

## Commands

**Build (ROS side)** — from the ROS 2 workspace root (`~/ros2_ws`), not this package directory:
```bash
colcon build --packages-select braid_tools
source install/setup.bash
```
Note: `--symlink-install` does not symlink `install(PROGRAMS)` scripts in ament_cmake — rebuild after editing anything in `nodes/`.

**Install the analysis library:**
```bash
cd braid_analysis  # or repo root, wherever setup.py is invoked from
python ./setup.py install
```
Runtime deps are in `requirements.txt` (pandas, h5py, matplotlib, pynumdiff, cvxpy, pyyaml, tables, bagpy, etc.) — `cvxpy` needs a solver like MOSEK for `flymath.get_convex_smoothed_course_and_ang_vel`.

**Run ROS nodes** (require a sourced workspace):
```bash
ros2 run braid_tools braid_ros_listener.py --braid-model-server-url=http://YOUR.IP:8397/
ros2 run braid_tools braid_realtime_plotter.py --config=plot_config/small_tunnel.yaml
ros2 run braid_tools braid_save_data_to_hdf5.py --ros-args -p data_directory:=~/data
ros2 run braid_tools braid_emulator.py   # fakes braid trajectories for testing, no hardware needed
ros2 run braid_tools braid_triggered_video_saver.py --config=video_config/triggered_video_saver.toml
ros2 launch braid_tools braid_listener.launch.py braid_url:=http://YOUR.IP:8397/
```

**Preprocess a raw `.braidz` dataset:**
```bash
python braid_analysis/preprocess_braidz.py DATA_DIR --length 50 --xdist 0.1
```
or the example-driven variant with a parameters yaml (see `example/analysis/preprocessing_parameters.yaml`):
```bash
python example/analysis/preprocess_braidz.py DATA_DIR preprocessing_parameters.yaml
```

**Compress an experiment directory for archival (strips 2D data, zips):**
```bash
bash scripts/compress_for_aws.sh /path/to/experiment_dir
```

There is no test suite, linter, or CI config in this repo currently.

## Architecture notes

### `braid_analysis` module structure
- `braid_filemanager.py` — all I/O: reading `.braidz` (a zip of gzipped CSVs: `data2d_distorted.csv.gz`, `kalman_estimates.csv.gz`, `cam_info.csv.gz`) into pandas DataFrames, plus saving/loading preprocessed HDF5. `load_filename_as_dataframe_3d`/`_2d` accept either a local path or a URL.
- `braid_slicing.py` — filters and slices trajectory DataFrames by `obj_id`: minimum length, distance travelled, spatial bounding volume, etc. Operates on the raw `obj_id` (per-file, reused across recordings) or `obj_id_unique` (globally unique, created by `assign_unique_id`).
- `flymath.py` — trajectory kinematics: course/heading angle computation, angular velocity smoothing (raw butterworth diff, or a slower but robust cvxpy-based total-variation smoother requiring MOSEK), and a modified Geometric Saccade Detector (multiprocessing-based) for saccade scoring/detection.
- `braid_2d_analysis.py` — connects 3D trajectories back to the raw 2D per-camera detections and trigger events: loading calibration/reprojection matrices, matching a 3D trajectory to camera-frame correspondences, and diagnostic plotting of trigger windows.
- `braid_analysis_plots.py` — general plotting helpers (trajectories, occupancy heatmaps, speed/heading histograms, arrowhead/wedge trajectory plots with time or stimulus color-coding).
- `preprocess_braidz.py` — CLI entry point: loads a `.braidz`, filters short/stationary trajectories (`braid_slicing`), computes course/angular velocity (`flymath`), and saves an HDF5 keyed as `DATA_<braidz_basename>`.

**Key convention:** `obj_id` is only unique within a single `.braidz` file/session. When combining data across multiple recordings, always use `obj_id_unique` (format: `"<braidz_basename>_<obj_id>"`, created via `braid_slicing.assign_unique_id`) instead — several functions in `braid_slicing.py` raise a `Warning` if they detect an `obj_id_unique` column is present but you passed `obj_id_key='obj_id'` anyway.

### Experiment directory convention
Scripts and examples assume a standard data directory layout per experiment:
```
experiment_dir/
├── *.braidz                # raw tracking data (zip of gzipped CSVs)
├── exp_code/                # experiment-specific metadata/code (optional)
├── *.bag, *.hdf5            # optional raw ROS recordings
└── preprocessed_data/       # created by preprocess_braidz.py, holds *_preprocessed.hdf
```
`scripts/compress_for_aws.sh` builds an archival copy of this layout with the 2D data stripped from the `.braidz` (2D data is only needed if you plan to retrack/rekalmanize post-hoc) and excludes `preprocessed_data/`.

### ROS message types
Custom messages in `msg/` (`FlydraMainbrainSuperPacket` → `FlydraMainbrainPacket[]` → `FlydraObject[]`) mirror the legacy `ros_flydra`/ROS 1 schema — field names are identical to the `main` branch, only the type names are CamelCase (required by rosidl). `braid_ros_listener.py` parses Braid's server-sent-event JSON stream and repacks each object update into this message hierarchy before publishing.

**ROS 1 bridge interop**: `braid_tools_mapping_rules.yaml` maps the ROS 1 snake_case message names to the ROS 2 CamelCase ones for ros1_bridge (which must be source-built with both branches' message sets). Keep the topic name `flydra_mainbrain/super_packets`, default QoS, and message field names identical across branches or the bridge breaks. See README "ROS 1 interoperability". `BraidTrigger.msg` is ROS 2-only (no mapping rule).

### Triggered video saver (`nodes/braid_triggered_video_saver.py`)
Listens for `braid_tools/BraidTrigger` on `braid_trigger`; on trigger, saves a pre+post-trigger MP4 from a strand-cam color camera plus a `metadata.hdf5` (trigger message + the triggered obj_id's tracking rows, same dtype as `braid_save_data_to_hdf5.py`) into a per-event directory under `<output_base_dir>/events/`. Configured by a braid-style toml in `video_config/` (installed to share; parsed with stdlib `tomllib`). Key design points:
- The node holds no frames: it launches `strand-cam` as a subprocess (requires the strand-braid deb) and drives its built-in post-trigger circular buffer over HTTP — POST `{"ToCamera": <CamArg>}` to `/callback`, e.g. `{"SetPostTriggerBufferSize": N}`, `"PostTrigger"`, `{"SetIsRecordingMp4": false}`; state comes from the `/strand-cam-events` SSE stream (`event: strand-cam` + one-line JSON `data:`). Localhost auth is bypassed via `--trusted-network`.
- The SSE state's `is_recording_mp4.path` does NOT reliably match the on-disk filename (~1 s offset observed); the node globs its private `.staging/` dir (strand-cam's `--data-dir`) and waits for the file size to stabilize instead. The encoder can keep draining its queue for many seconds after the stop command — hence the retries around `show-timestamps` and the generous (30 s) tracking-buffer retention so rows survive until the finalize thread snapshots them.
- Clock domains: the tracking window is evaluated on `acquire_stamp`; the MP4's per-frame MISP timestamps (dump with `show-timestamps --output csv`) share that clock only when strand-cam runs with `--braid-url`.
- Overlapping triggers are dropped (strand-cam has a single post-trigger buffer); strand-cam dying makes the node exit (fail-fast).
- Encoding (verified 2026-07-06, strand-braid 1.0.0-rc.5): the shipped toml uses `mp4_codec = { Ffmpeg = { codec = "h264_nvenc" } }` + `mp4_max_framerate = "Unlimited"` — full-rate color, zero drops (105 fps × 60 s tested); requires system ffmpeg. rc.5's colorspace fix is what lets Bayer frames reach ffmpeg (rc.4 errored). The built-in `"H264Nvenc"` codec still fails on Blackwell GPUs (upstream issue #29, SDK-11 bindings; `is_nvenc_functioning: false`); rc.5 also made mp4-writer errors non-fatal (no more crashes). Fallbacks in `NVENC_UPGRADE_TODO.md`.
- Future work planned: overlaying reprojected 3D tracks onto saved frames (`strand_scripts/calibrate_aux_camera.py` provides the calibration + `project_points()` helper; `braid_2d_analysis.reproject` works with the emitted `P_linear`).

### Standalone strand-cam tools (`strand_scripts/`)
ROS-free, pandas-using scripts that drive strand-cam directly (NOT installed by ament; not part of `braid_analysis`). `collect_led_calibration_data.py` records braid 3D (direct SSE from the model server, no ROS) + aux-camera 2D LED detections (strand-cam point detection → flytrax CSV) into per-session dataset dirs with a merged 3D↔2D correspondence file; `led_calibration_merge.py` is the importable merge module (`--selftest` runs synthetic checks). Hard-won facts baked into these scripts:
- flytrax CSV header is `#`-commented **YAML** (`created_at` key; absolute time = created_at + `time_microseconds`·1e-6); columns end with empty `led_1..3`.
- `SetObjDetectionConfig` takes a YAML string of `ImPtDetectCfg`; merge over the live `im_pt_detect_cfg` from SSE state before sending. Polarity values are `DetectLight`/`DetectDark`/`DetectAbsDiff` (NOT "DetectBright"). `SetIsSavingObjDetectionCsv` takes `{"Saving": null}` / `"NotSaving"`. `TakeCurrentImageAsBackground` is a bare (non-ToCamera) callback string.
- strand-cam has **no HTTP snapshot route** — the reference image is grabbed by recording a ~0.6 s MP4 and extracting a frame with ffmpeg.
- Clock robustness: braid rows are timestamped on the braid PC's triggerbox clock, flytrax on the local clock; the merge fits each clock against the exact frame counters (de-jitter), estimates the offset by speed-profile cross-correlation, then frame-locks by scanning candidate integer frame offsets for lowest DLT reprojection error. Frame locking is gated on the two fitted frame periods agreeing (≤0.5%).
- Point detection at full 1920x1200 color measured only ~11.6 fps on this machine during testing (free-run; encoder/detector load) — check `measured_fps` and the flytrax row rate when data seems sparse. (On the real triggerbox-synced rig it ran at the full 100 fps.)
- `calibrate_aux_camera.py` fits K + plumb_bob distortion + R/t from merged.csv (single-view volumetric resectioning, no checkerboard). RANSAC-initialized DLT (8-point samples, `--ransac-px`, default 8 px) because LED reflections off the arena can contaminate ~45% of correspondences with spatially coherent bogus streaks — a plain all-points DLT + soft_l1 + MAD rejection collapses at that level (44%-contaminated real dataset: fx=67 nonsense before, 2.6 px median after). scipy `least_squares` (soft_l1, bounded) chosen over `cv2.calibrateCamera` (single non-planar view requires `CALIB_USE_INTRINSIC_GUESS` and has no robust loss); k3 fixed to 0 by default because the LED only covers part of the sensor (see `valid_pixel_region` in the yaml — computed from inliers only, so reflections don't inflate it). Emits `camera_calibration.yaml` (`P_linear` matches the `reproject` convention) + diagnostics PNG; `load_camera_calibration()`/`project_points()` are the API for the overlay script. Reflection-heavy real dataset: 57% RANSAC consensus, 2.6 px median.

### Calibration alignment tool (`braid_calibration_mcsc_alignment/`)
Automates flydra's `flydra_analysis_calibration_align_gui` "step 5": aligning an unaligned MCSC calibration (`<multi_camera_reconstructor>` XML) to arena geometry using a wand-tracing braidz. `align_core.py` is the importable GUI-free core (XML/stimxml/braidz I/O, similarity algebra, robust fit, frustum geometry; `--selftest` runs synthetic checks); `align_calibration.py` is the CLI + pyvista/pyvistaqt GUI (see its `--help`; `--no-gui` does headless autofit + save + PNG snapshot). Key facts:
- Convention: points `X' = sR X + t` (`M = [[sR, t],[0,1]]`), cameras `P' = P·M⁻¹` renormalized to `P'[2,3]=1`; `non_linear_parameters` are passed through as verbatim text. Reprojection invariance, XML round-trip, and similarity checks run on every save.
- Autofit minimizes robust (soft_l1 + MAD-trimmed) point-to-box-**surface** distance so the interior axis-arrow strokes don't bias the fit; multi-starts over the box symmetry group **including reflections** — the example dataset's MCSC calibration (`example_calibration_data/`, the big wind tunnel) really is mirrored (det = −1). Real-data result: median 4.1 mm to surface, scale 0.671.
- Geometry alone cannot resolve axis flips/90° hops when the arena cross-section is square — the user must check the drawn arrows against the GUI's +X/+Y/+Z arrows and use the Flip/+90° controls before saving.
- Camera frusta are computed decomposition-free (P null space + per-ray cheirality sign fix) because `cv2.decomposeProjectionMatrix`-style det-forcing points frusta backwards for mirrored calibrations.
- `ALIGN_GUI_SCREENSHOT=/path.png` env var makes the GUI self-screenshot and quit after 4 s (automated smoke test; `QWidget.grab()` can't capture the GL viewport — the hook also saves a `-viewport.png` via VTK).
- pip installs on this box need `--user --break-system-packages` (PEP 668); pyvista/pyvistaqt/vtk installed that way.

### Example/demo layout
`example/` contains a full worked demo (`20241203_led_demo/`) and Jupyter notebooks (`example/analysis/*.ipynb`) that exercise the preprocessing → diagnostic-plotting pipeline end-to-end — useful as a reference when adding new analysis functions.
