# Braid Tools

* Helpful nodes for interfacing Braid with ROS 2.
* Basic analysis, slicing, and plotting functions.

This branch (`ros2`) targets **ROS 2 Jazzy** on Ubuntu 24.04. The `main` branch contains the original ROS 1 (catkin) version.

# Building

From your ROS 2 workspace root (e.g. `~/ros2_ws`):

```bash
colcon build --packages-select braid_tools
source install/setup.bash
```

Python dependencies for the nodes come from apt (`python3-requests`, `python3-h5py`, `python3-matplotlib`, `python3-yaml`, `python3-numpy`; `rosdep install` will pick them up from package.xml). The emulator optionally uses `pynumdiff` for velocity estimation, which is pip-only — on Ubuntu 24.04 install it in a venv or with `pip install --break-system-packages pynumdiff`; without it the emulator falls back to `np.gradient`.

# ROS nodes

### braid_ros_listener.py

Node that listens to an active Braid instance and streams the data to a ROS topic: `flydra_mainbrain/super_packets`. To run:
* `ros2 run braid_tools braid_ros_listener.py --braid-model-server-url=http://YOUR.IP.ADDRESS:8397/`

Or via launch file (defaults to the lab Braid server):
* `ros2 launch braid_tools braid_listener.launch.py braid_url:=http://YOUR.IP.ADDRESS:8397/`

### braid_emulator.py

Simulates a moving trajectory and outputs in the same format as the braid_ros_listener. Helpful for testing hardware without needing real trajectory data.
* `ros2 run braid_tools braid_emulator.py --fps 100`

### braid_realtime_plotter.py

Allows you to visualize where in space tracked objects are. Helpful for trouble shooting orientations of calibrations and testing tracking. Only supports a single object (first object). Uses a yaml file to set plotting dimensions, modify to fit your arena. To run:
* `ros2 run braid_tools braid_realtime_plotter.py --config=/path/to/plot_config/small_tunnel.yaml`

If `--config` is omitted, the installed `plot_config/small_tunnel.yaml` is used. There is also a combined launch file:
* `ros2 launch braid_tools braid_listener_and_live_plotter.launch.py plot_config:=/path/to/big_tunnel.yaml`

### braid_save_data_to_hdf5.py

ROS node for saving `flydra_mainbrain/super_packets` to an hdf5 file with buffering etc.
* `ros2 run braid_tools braid_save_data_to_hdf5.py --ros-args -p data_directory:=/path/to/save/dir`

The save directory can be given either with the `data_directory` ROS parameter (shown above) or the `--home-directory` CLI argument; defaults to `~/Desktop/temp`.

### braid_triggered_video_saver.py

See [TRIGGERED_VIDEO_QUICKSTART.md](TRIGGERED_VIDEO_QUICKSTART.md) for a 5-minute hands-on test, and [NVENC_UPGRADE_TODO.md](NVENC_UPGRADE_TODO.md) for the pending GPU-encoding upgrade (strand-braid issue #29).

Saves a short video clip from a [strand-cam](https://github.com/strawlab/strand-braid) color camera whenever a trigger message arrives, together with the braid 3D tracking data for the triggered object. Requires the strand-braid `.deb` to be installed (provides the `strand-cam` and `show-timestamps` executables).

The node launches strand-cam as a managed subprocess and arms its built-in post-trigger circular frame buffer (sized to `pre_trigger_seconds`, nominally 1 s). When a `braid_tools/BraidTrigger` message with `trigger: true` arrives on the `braid_trigger` topic, strand-cam flushes the buffered pre-trigger frames into a new MP4 and keeps recording for `post_trigger_seconds` (nominally 5 s). The node holds no frames itself, so the video is hardware-timestamped by strand-cam: each frame carries a MISP microsecond timestamp, which is on the braid triggerbox clock (the same clock as `acquire_stamp`) when strand-cam is connected to braid via `braid_url`.

* `ros2 run braid_tools braid_triggered_video_saver.py --config=/path/to/triggered_video_saver.toml`
* `ros2 launch braid_tools braid_triggered_video_saver.launch.py config:=/path/to/triggered_video_saver.toml`

If `--config` is omitted, the installed `video_config/triggered_video_saver.toml` is used. The config follows the braid/strand-cam toml conventions (a `[[cameras]]` block naming the camera plus a `[triggered_video_saver]` section); `output_base_dir`, `pre_trigger_seconds` and `post_trigger_seconds` can also be overridden as ROS parameters (`--ros-args -p post_trigger_seconds:=10.0`).

The trigger message is:
```
builtin_interfaces/Time stamp   # trigger time (unix epoch domain); leave zero to use receive time
bool trigger                    # ignored when false
uint32 obj_id                   # braid obj_id (matches FlydraObject.obj_id)
string metadata                 # optional free-form JSON, copied verbatim into the event metadata
```
For example: `ros2 topic pub --once /braid_trigger braid_tools/msg/BraidTrigger "{trigger: true, obj_id: 1234, metadata: '{\"reason\": \"entered volume\"}'}"`

Each trigger event gets its own directory:
```
<output_base_dir>/
├── strand-cam.log                # strand-cam output for the session
├── .staging/                     # strand-cam data dir; in-progress MP4s live here
└── events/
    └── 20260702_143210.123456_obj42/
        ├── movie..._<CAMNAME>.mp4    # pre+post trigger video (strand-cam's native name)
        ├── metadata.hdf5             # trigger info + tracking rows (see below)
        └── frame_timestamps.csv      # per-frame MISP timestamps (from show-timestamps)
```
`metadata.hdf5` stores the full trigger message, camera name, buffer/window settings, a snapshot of the toml config, and a `tracking_rows` dataset containing the triggered `obj_id`'s rows from `flydra_mainbrain/super_packets` spanning the pre+post window (same fields as `braid_save_data_to_hdf5.py`). Read it with `h5py` or `pandas.DataFrame(h5py.File(f)['tracking_rows'][:])`.

Notes:
* Triggers arriving while an event is in progress are dropped with a warning (strand-cam has a single post-trigger buffer).
* Set `fps` in the toml to the camera's actual framerate — it sizes the pre-trigger buffer in frames. The node warns if the measured fps deviates by more than 20 %.
* The shipped config uses GPU encoding via the system ffmpeg (`mp4_codec = { Ffmpeg = { codec = "h264_nvenc" } }`, requires ffmpeg installed): verified full-rate color with zero dropped frames (105 fps sustained, strand-braid ≥ 1.0.0-rc.5 — earlier versions can't feed Bayer color frames to ffmpeg). The built-in `H264Nvenc` codec still fails on Blackwell GPUs (strand-braid issue #29); the built-in `H264OpenH264` software encoder only sustains ~37 fps at 1920×1200 (cap `mp4_max_framerate = "Fps30"` if you must use it). See `NVENC_UPGRADE_TODO.md` for details and fallbacks.
* To test without braid: run `ros2 run braid_tools braid_emulator.py` and publish a trigger for a live obj_id.

# Standalone strand-cam tools (`strand_scripts/`)

ROS-free utilities that drive a local [strand-cam](https://github.com/strawlab/strand-braid) camera directly (require the strand-braid `.deb`). **Hands-on workflow: see [strand_scripts/README.md](strand_scripts/README.md)** (collect → check merge → calibrate).

### collect_led_calibration_data.py

Collects a calibration dataset for an auxiliary camera that shares the triggerbox with a braid array running on another machine: move an LED around the shared volume while the script records braid 3D positions (from the braid model server's HTTP stream) and 2D LED pixel detections (strand-cam point detection → flytrax CSV) simultaneously.

```bash
python3 strand_scripts/collect_led_calibration_data.py                      # default co-located toml
python3 strand_scripts/collect_led_calibration_data.py --config my.toml --duration 120
python3 strand_scripts/collect_led_calibration_data.py --merge-only DATASET_DIR   # re-run the merge
```

The toml (`strand_scripts/led_calibration.toml`) uses braid's exact `[[cameras]]` + `[cameras.point_detection_config]` format, so the detection table can be copied from a braid config. The script launches strand-cam, applies the detection config, and prints the browser UI URL — check the live image there, press Enter to record, Ctrl-C to stop. Each session writes its own dataset directory: raw `braid_3d.csv` + strand-cam's flytrax CSV + `reference.png` + `metadata.yaml`, plus **`merged.csv`** with per-frame 3D↔2D correspondences and an `alignment_diagnostics.png` to eyeball.

Row matching is robust to unsynchronized PC clocks (run ptpd or don't): the clock offset is estimated by cross-correlating the LED's speed profile in both streams, and then — because both cameras count the same triggerbox pulses — the exact integer frame offset is found by testing candidate offsets for geometric consistency (DLT reprojection error), giving jitter-free frame-locked correspondences. Frame locking auto-disables (falling back to nearest-time matching, with a warning) if the two streams' fitted frame rates disagree, i.e. if the cameras aren't actually synced. `python3 strand_scripts/led_calibration_merge.py --selftest` exercises the merge on synthetic data.

Wave the LED with continuously varying speed and cover the whole field of view; a stationary LED cannot be aligned (the merge warns via low `peak_correlation`).

**Tuning detection:** strand-cam's browser UI is the tuning tool (live image with detections overlaid + all detection settings). Tune while the collection script waits at the press-Enter prompt, then dump the tuned values as a paste-ready toml block with `python3 strand_scripts/dump_detection_config.py`. Update the toml afterwards — keys present in the toml override the UI-tuned (persisted) values on the next launch.

### calibrate_aux_camera.py

Fits the auxiliary camera's calibration — intrinsics, lens distortion (OpenCV plumb_bob), and extrinsics (braid world → camera) — from a collected LED dataset. No checkerboard needed: the volumetric LED trajectory is a single-view resectioning target, and with thousands of correspondences the nonlinear distortion is well constrained (visualization-grade; a real dataset fit to 2.2 px median at 1920×1200).

```bash
python3 strand_scripts/calibrate_aux_camera.py DATASET_DIR          # after collection+merge
python3 strand_scripts/calibrate_aux_camera.py --selftest
```

Pipeline: normalized DLT → decompose → robust nonlinear refinement (scipy `least_squares`, soft-L1 loss, bounded parameters) with iterative outlier rejection; `--cross-check` validates against `cv2.calibrateCamera`. Writes into the dataset dir:
* `camera_calibration.yaml` — K, distortion, R/t, and `P_linear` (drop-in for `braid_analysis.braid_2d_analysis.reproject`), plus fit stats and the valid pixel region (the distortion model is only constrained where the LED actually went; `k3` is fixed to 0 by default to keep extrapolation sane — `--k3` frees it).
* `calibration_diagnostics.png` — residual quivers (linear vs full model), histograms, and the reprojected trajectory overlaid on `reference.png`.

The future overlay script consumes the yaml via `calibrate_aux_camera.load_camera_calibration()` and `project_points()` (full distortion model, behind-camera points masked).

# ROS 1 interoperability

## Option 1: topic relay (recommended, simple)

A lightweight relay streams ROS 1 topics to ROS 2 over HTTP the same way Braid streams tracking data — no ros1_bridge, no Docker.

* **On the ROS 1 machine** (`main` branch of this repo): run the relay server with a yaml listing the topics to relay (see `relay_config/relay_topics.yaml`):
  ```bash
  rosrun braid_tools topic_relay_server.py --config $(rospack find braid_tools)/relay_config/relay_topics.yaml
  ```
* **On the ROS 2 machine** (this branch): run the client pointed at the ROS 1 machine:
  ```bash
  ros2 run braid_tools topic_relay_client.py --url http://ROS1.MACHINE.IP:8398/
  ```

The client needs no configuration — the server announces which topics and message types it relays, and the client republishes them under the same topic names. Message types are auto-detected on the ROS 1 side; only types whose ROS 2 name differs from `pkg/msg/Name` need an entry in the yaml's `type_map` (the braid_tools messages are pre-mapped).

How it works: the server subscribes to the configured topics and re-serves each message as JSON over an HTTP server-sent-event stream (default port 8398), the same pattern Braid uses for its tracking data. It supports multiple simultaneous clients, drops oldest messages if a client reads too slowly, and sends a heartbeat every 2 s when idle so clients can detect a dead connection. The client reconnects with exponential backoff (max 10 s) if the server restarts, and resumes publishing automatically.

Limitations: every message is JSON-encoded, which is fine for telemetry-sized data (tracking packets, triggers, floats) but unsuitable for images or point clouds. One direction only: ROS 1 → ROS 2.

**Smoke test.** On the ROS 1 machine, with the server running:
```bash
curl -N http://localhost:8398/events
```
You should see a `ros1_relay_hello` event listing the relayed topics, then a stream of `ros1_relay` events while data flows (or `ros1_relay_heartbeat` events every 2 s if nothing is publishing). On the ROS 2 machine, `ros2 topic list` should show the relayed topics shortly after the client connects, and `ros2 topic hz /flydra_mainbrain/super_packets` should match the rate seen by `rostopic hz` on the ROS 1 side.

## Option 2: ros1_bridge

For full bidirectional bridging, `ros1_bridge` works but must be source-built. The following matter on this package's side:

* The package name (`braid_tools`) and message field names are identical on both branches, but ROS 2 requires CamelCase message names (`flydra_object` → `FlydraObject`, etc.). The mapping is declared in `braid_tools_mapping_rules.yaml`, which is installed to the package share directory and exported for the bridge.
* Custom messages always require a **source-built** ros1_bridge whose ROS 1 underlay includes the `main` branch build of this package and whose ROS 2 underlay includes the `ros2` branch build.
* There is no official ros1_bridge release for Jazzy/24.04. The usual setup is a Noetic + Humble bridge host (or container); Jazzy nodes interoperate with the Humble side over DDS since both build identical message definitions from this repo.
* The nodes use default QoS (reliable/volatile) and the plain topic name `flydra_mainbrain/super_packets`, both of which the bridge handles automatically.
* ROS 1 `time` fields map automatically to `builtin_interfaces/Time`.

# Analysis

The `braid_analysis` Python library is ROS-independent and unchanged from the `main` branch. From inside this directory, run `python ./setup.py install` to install the analysis tools.

See `example` for demos.
