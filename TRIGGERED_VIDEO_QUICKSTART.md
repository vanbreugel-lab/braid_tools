# Triggered video saver — 5-minute quickstart

Records a color MP4 (1 s pre-trigger + 5 s post-trigger) from the strand-cam camera
whenever a trigger message arrives, plus the braid tracking data for the triggered
object. No braid server needed for this test — the emulator fakes the tracking stream
and the camera free-runs.

Config used below: `video_config/triggered_video_saver.toml`
(camera `Basler-41942900`, OpenH264 color @ 30 fps cap — see `NVENC_UPGRADE_TODO.md`
for why the framerate is capped and how to lift it later).

## 0. Build once

```bash
cd ~/ros2_ws
colcon build --packages-select braid_tools
source install/setup.bash        # in every terminal below
```

## 1. Terminal A — fake tracking data

```bash
ros2 run braid_tools braid_emulator.py
```

## 2. Terminal B — the video saver (launches strand-cam itself)

```bash
ros2 run braid_tools braid_triggered_video_saver.py \
  --config ~/ros2_ws/src/braid_tools/video_config/triggered_video_saver.toml
```

Pointing `--config` at the repo copy means config edits take effect on restart, no
rebuild needed. Wait for: `post-trigger buffer armed: ... frames`.

Note: free-running, the camera does ~156 fps but the toml says `fps = 100.0` (the
braid-synced rate), so the node will warn that the pre-trigger buffer spans <1 s.
Harmless for a test; set `fps = 156.0` if you want the exact pre-trigger duration.

## 3. Terminal C — fire a trigger

Grab a currently-alive obj_id and trigger on it in one go (emulated objects live
only a few seconds, so sample-and-fire quickly):

```bash
OBJ=$(ros2 topic echo --once /flydra_mainbrain/super_packets --field packets \
      | grep -o 'obj_id=[0-9]*' | head -1 | cut -d= -f2) && \
ros2 topic pub --once /braid_trigger braid_tools/msg/BraidTrigger \
  "{trigger: true, obj_id: $OBJ, metadata: '{\"reason\": \"quickstart test\"}'}"
```

Terminal B logs `trigger received ...` and, ~10 s later, `event complete: ...`.

## 4. Look at the result

```bash
ls ~/Desktop/triggered_videos/events/            # one directory per trigger event
```

Each event dir contains the MP4 (play it with any player), `frame_timestamps.csv`
(per-frame hardware timestamps), and `metadata.hdf5`:

```python
import h5py, pandas as pd
f = h5py.File('metadata.hdf5')
print(dict(f.attrs))                             # trigger info, camera, settings
df = pd.DataFrame(f['tracking_rows'][:])         # 3D tracking for the triggered obj_id
```

If the emulated object died between sampling and triggering you may get 0 tracking
rows (the `error` attr says so) — just fire another trigger. A second trigger sent
while an event is still recording is dropped with a warning (single buffer).

## 5. Real experiment differences

In the toml: set `braid_url` to your mainbrain (e.g. `http://134.197.37.229:3333/`)
so frames are triggerbox-synced, set `fps` to the braid framerate, and point
`camera_settings_filename` at your `.pfs` file. Run the real
`braid_ros_listener.py` instead of the emulator. Your experiment code publishes
`braid_tools/BraidTrigger` messages on `braid_trigger`.
