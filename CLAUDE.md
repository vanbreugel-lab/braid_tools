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

**ROS 1 bridge interop**: `braid_tools_mapping_rules.yaml` maps the ROS 1 snake_case message names to the ROS 2 CamelCase ones for ros1_bridge (which must be source-built with both branches' message sets). Keep the topic name `flydra_mainbrain/super_packets`, default QoS, and message field names identical across branches or the bridge breaks. See README "ROS 1 interoperability".

### Example/demo layout
`example/` contains a full worked demo (`20241203_led_demo/`) and Jupyter notebooks (`example/analysis/*.ipynb`) that exercise the preprocessing → diagnostic-plotting pipeline end-to-end — useful as a reference when adding new analysis functions.
