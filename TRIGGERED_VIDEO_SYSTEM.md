# Operating the triggered aux-camera video system (ROS 1 + ROS 2)

The ROS 1 machine runs braid, the triggerbox, and the volume-trigger logic
(`windtunnel_optotrigger_experiments`); this ROS 2 machine runs the strand-cam
auxiliary color camera and saves a pre+post-trigger video + tracking metadata
for every trigger event. Everything crosses between the machines over plain
HTTP — no ros1_bridge.

```
ROS 1 machine (braid + triggerbox)                 ROS 2 machine (this one)
──────────────────────────────────                 ─────────────────────────────────
braid ──── model server :8397 ───── HTTP ────────► braid_ros_listener.py
  │                                                      │ /flydra_mainbrain/super_packets
  ▼ (ROS 1 topic)                                        ▼
wind_tunnel_openloop_trigger_in_volume.py          braid_triggered_video_saver.py ──► strand-cam
  │ /braid_trigger_topic (Float64MultiArray)             ▲ /braid_trigger (BraidTrigger)     │
  ▼                                                      │                                   ▼
topic_relay_server.py :8398 ─────── HTTP ────────► topic_relay_client.py ──► braid_trigger_adapter.py
                                                                              events/<timestamp>_obj<id>/
                                                                                ├── movie....mp4
                                                                                ├── metadata.hdf5
                                                                                └── frame_timestamps.csv
triggerbox ── 100 fps hardware trigger line ─────────────────────────────────► aux camera
```

## One-time setup

- **ROS 1 machine** (`main` branch of braid_tools): add the trigger topic to the
  relay config so it is streamed to ROS 2 — one line in
  `relay_config/relay_topics.yaml`:
  ```yaml
  topics:
    - /flydra_mainbrain/super_packets
    - /braid_trigger_topic          # <-- add this (name from the trigger node's config yaml)
  ```
  No `type_map` entry needed (`std_msgs/Float64MultiArray` maps automatically).
- **ROS 2 machine**: `colcon build --packages-select braid_tools` once (and after
  any edit under `nodes/`); check `video_config/triggered_video_saver.toml`
  (camera name, `.pfs` with TriggerMode enabled, `output_base_dir`,
  pre/post seconds, `braid_url`).
- The aux camera must be wired to the same triggerbox as the braid cameras.

## Run an experiment

**ROS 1 machine** (in this order):
1. Braid running (which also drives the triggerbox → the aux camera gets frames).
2. Relay server:
   ```bash
   rosrun braid_tools topic_relay_server.py --config $(rospack find braid_tools)/relay_config/relay_topics.yaml
   ```
3. Trigger node:
   ```bash
   rosrun windtunnel_optotrigger_experiments wind_tunnel_openloop_trigger_in_volume.py --config YOUR_TRIGGER_CONFIG.yaml
   ```

**ROS 2 machine** — one command starts all four nodes (listener, relay client,
trigger adapter, video saver + its strand-cam):
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch braid_tools braid_triggered_video_pipeline.launch.py \
    braid_url:=http://BRAID.MACHINE.IP:8397/ \
    relay_url:=http://BRAID.MACHINE.IP:8398/
```
(Defaults point at the lab braid machine; `video_config:=` and
`trigger_topic:=` override the toml and topic name if needed.)

Wait for `post-trigger buffer armed: ...` in the output — the system is live.
Each trigger then logs `trigger forwarded` → `trigger received` →
(~10 s later) `event complete`, and a new directory appears under
`<output_base_dir>/events/` containing the color MP4 (1 s before the trigger +
`post_trigger_seconds` after), `metadata.hdf5` (full trigger info including the
arduino stimulus values, plus the triggered object's 3D tracking rows), and
per-frame hardware timestamps.

## Verification ladder (first bring-up / debugging)

Run these on the ROS 2 machine, in order — each step isolates one link:

1. `curl -sN http://BRAID.MACHINE.IP:8397/events | head -3` — braid tracking stream reachable.
2. `curl -sN http://BRAID.MACHINE.IP:8398/events | head -3` — relay reachable; the
   `ros1_relay_hello` must list `/braid_trigger_topic`. If it doesn't, the topic
   isn't in `relay_topics.yaml` or the trigger node hasn't published yet (the
   server resolves types on first publish — fire one test trigger).
3. `ros2 topic hz /flydra_mainbrain/super_packets` — tracking flowing into ROS 2.
4. `ros2 topic echo /braid_trigger_topic` — raw relayed trigger arrays arriving.
5. `ros2 topic echo /braid_trigger` — adapted `BraidTrigger` (obj_id, stamp, metadata).
6. A real trigger → event directory complete.

A bench test without the ROS 1 machine at all: `TRIGGERED_VIDEO_QUICKSTART.md`
(braid emulator + manual `ros2 topic pub`).

## Gotchas

- **Refractory time**: set the trigger node's `refractory_time` ≥ the saver's
  pre+post window (default 1 + 5 = 6 s). strand-cam has a single post-trigger
  buffer, so triggers arriving while an event is recording are dropped (with a
  warning) by design.
- **Clocks**: the trigger's timestamp is the braid machine's wall clock — the
  same clock domain as the tracking timestamps, so the pre/post tracking window
  is correct without any cross-machine clock sync (ptpd not required for this).
- **Ports**: 8397 (braid model server) and 8398 (relay) must be reachable from
  this machine — check firewalls if the curls in the ladder fail.
- **Message layout**: the adapter expects the trigger node's
  `data = [flag, obj_id, framenumber, t_wall] + stimulus_values`. If the ROS 1
  node's published layout changes, update `nodes/braid_trigger_adapter.py`.
- **Camera settings**: the `.pfs` in the video toml must enable external
  triggering (TriggerMode=On); with the triggerbox idle, strand-cam exits with a
  pylon "Grab timed out" error at startup. Exposure in the LED-tuned pfs may be
  too dark for demo video — keep a video-tuned copy.
- **USB buffer**: `usbfs_memory_mb` should be ≥ 1000 for sustained 100 fps color
  (`usbcore.usbfs_memory_mb=1024` on the kernel cmdline).
